"""Genetic-programming (GP) repair baseline.

A (mu + lambda) evolutionary search over RASP source programs. The population is seeded
with the buggy mutant and random single-step mutations of it; each generation produces
up to `population_size` offspring by tournament selection + one random mutation, and
the best `population_size` programs (by fitness) survive. Every generated program counts
toward `budget`, including duplicates. See `common` for fitness and evaluation.

v1 (paper) settings: budget 1000, population 16, tournament 3, seed 42.
Ported from `experiments/gp_baseline/gp_core.py` with identical behaviour (RNG calls
and their order, history records, status strings, result keys); it is pinned by
`tests/test_golden_search.py`.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rasping import programs
from rasping.baselines.common import (
    apply_mutation_once,
    compute_max_length,
    evaluate_source,
    hash_source,
    history_entry,
    load_splits,
)
from rasping.config import DATA_DIR

logger = logging.getLogger(__name__)

#: Hyperparameters of the v1 (TMLR) runs.
DEFAULTS = dict(budget=1000, population_size=16, tournament_k=3, seed=42)


@dataclass
class Candidate:
    source: str
    train_acc: float
    test_acc: float = None


def run_gp(
    base_source: str,
    program: str,
    data_dir: Path = DATA_DIR,
    budget: int = DEFAULTS["budget"],
    population_size: int = DEFAULTS["population_size"],
    tournament_k: int = DEFAULTS["tournament_k"],
    seed: int = DEFAULTS["seed"],
    log_every: int = 25,
    eval_timeout: float = 30.0,
) -> Dict[str, Any]:
    """Run GP from `base_source` (a mutant of `program`). Returns the v1 result dict
    (without the job metadata that `run_gp_for_mutant` adds)."""
    rng = random.Random(seed)
    program = programs.canonical(program)

    train_data, val_data, test_data = load_splits(program, data_dir)
    fitness_data = train_data + val_data  # F5: fitness uses train+val
    max_length = compute_max_length(program, data_dir)

    seen: set[str] = set()
    cache: Dict[str, Tuple[float, float]] = {}
    num_programs = 0
    num_duplicates = 0
    num_compile_failures = 0

    def get_or_eval(src: str) -> Optional[Tuple[float, float]]:
        nonlocal num_compile_failures, num_duplicates
        h = hash_source(src)
        if h in cache:
            num_duplicates += 1
            return cache[h]
        eval_start = time.time()
        logger.debug(f"Evaluating candidate {h[:8]}... (program #{num_programs})")
        try:
            train_acc = evaluate_source(src, program, max_length, fitness_data, timeout=eval_timeout)
            test_acc = evaluate_source(src, program, max_length, test_data, timeout=eval_timeout)
            logger.debug(f"Candidate {h[:8]}... evaluated in {time.time() - eval_start:.2f}s: "
                         f"train={train_acc:.4f}, test={test_acc:.4f}")
        except Exception as e:  # unreachable: evaluate_source never raises
            num_compile_failures += 1
            logger.debug(f"Compile/eval failed for candidate {h[:8]}... "
                         f"after {time.time() - eval_start:.2f}s: {e}")
            return None
        cache[h] = (train_acc, test_acc)
        if log_every and num_programs % max(1, log_every) == 0:
            logger.info(f"Progress: programs={num_programs}/{budget} | dup={num_duplicates} "
                        f"| fails={num_compile_failures}")
        return train_acc, test_acc

    population: List[Candidate] = []
    logger.info(f"Starting GP search for program={program} with budget={budget}, "
                f"pop={population_size}, k={tournament_k}, seed={seed}")
    program_history: List[Dict[str, Any]] = []

    best_train = -1.0
    best_test = -1.0

    # The base (buggy) program counts towards the budget.
    num_programs += 1
    base_eval = get_or_eval(base_source)
    if base_eval is not None:
        best_train, best_test = base_eval
        population.append(Candidate(base_source, base_eval[0], base_eval[1]))
        seen.add(hash_source(base_source))
        logger.info(f"Seeded base candidate: train={base_eval[0]:.4f} test={base_eval[1]:.4f}")
        status, train_acc, test_acc = "ok", base_eval[0], base_eval[1]
    else:
        status, train_acc, test_acc = "compile_fail", None, None

    generation = 0
    program_history.append(history_entry(num_programs, "generation", generation, status,
                                         train_acc, test_acc, best_train, best_test))
    generation += 1

    # Fill the population with random single-step mutations of the base program.
    attempts = 0
    while len(population) < population_size and attempts < population_size * 20 \
            and num_programs < budget:
        attempts += 1
        mutated = apply_mutation_once(base_source, rng)
        if mutated is None:
            continue
        num_programs += 1  # counted regardless of duplicate / compile status
        h = hash_source(mutated)
        if h in seen:
            logger.debug(f"Duplicate candidate skipped: {h}")
            num_duplicates += 1
            status = "duplicate"
            eval_result = None
        else:
            eval_result = get_or_eval(mutated)
            status = "compile_fail" if eval_result is None else "ok"

        if status == "ok" and eval_result is not None:
            train_acc, test_acc = eval_result
            # NB: ties on fitness are broken by *test* accuracy
            if train_acc > best_train or (train_acc == best_train and test_acc > best_test):
                best_train = train_acc
                best_test = test_acc
            seen.add(h)
            population.append(Candidate(mutated, train_acc, test_acc))
            logger.debug(f"Seeded mutated candidate {h}: train={train_acc:.4f} test={test_acc:.4f}")
        else:
            train_acc = eval_result[0] if eval_result is not None else None
            test_acc = eval_result[1] if eval_result is not None else None

        program_history.append(history_entry(num_programs, "generation", generation, status,
                                             train_acc, test_acc, best_train, best_test))

    if not population:
        return {
            "status": "failed_to_initialize",
            "num_programs": num_programs,
            "program_history": program_history,
        }

    # Best-by-fitness of the initial population (first maximum; no test tie-break here).
    best_by_train = max(population, key=lambda c: c.train_acc)
    best_train = best_by_train.train_acc
    best_test = best_by_train.test_acc
    logger.info(f"Initial best: train={best_by_train.train_acc:.4f}")

    generation += 1  # NB: incremented again at the top of the loop (generation 2 is skipped)

    if best_train >= 1.0:
        # Re-evaluates the test accuracy (not counted toward the budget).
        test_acc = evaluate_source(best_by_train.source, program, max_length, test_data,
                                   timeout=eval_timeout)
        logger.info(f"Early stop: perfect train achieved after {num_programs} programs "
                    f"(dup={num_duplicates}, fails={num_compile_failures}). Test={test_acc:.4f}")
        return {
            "status": "success_early",
            "num_programs": num_programs,
            "num_duplicates": num_duplicates,
            "num_compile_failures": num_compile_failures,
            "train_acc": best_train,
            "test_acc": test_acc,
            "best_source": best_by_train.source,
            "program_history": program_history,
        }

    def tournament_select(pop: List[Candidate]) -> Candidate:
        contenders = rng.sample(pop, k=min(tournament_k, len(pop)))
        return max(contenders, key=lambda c: c.train_acc)

    while num_programs < budget:
        offspring: List[Candidate] = []
        generation += 1
        logger.info(f"Generation {generation} start: programs={num_programs}/{budget}, "
                    f"dup={num_duplicates}, fails={num_compile_failures}")

        gen_attempts = 0
        while len(offspring) < population_size and num_programs < budget \
                and gen_attempts < population_size * 50:
            gen_attempts += 1
            parent = tournament_select(population)
            mutated = apply_mutation_once(parent.source, rng)
            if mutated is None:
                continue
            num_programs += 1
            h = hash_source(mutated)
            if h in seen:
                logger.debug(f"Duplicate candidate skipped: {h}")
                num_duplicates += 1
                status = "duplicate"
                train_acc = None
                test_acc = None
                if h in cache:
                    train_acc, test_acc = cache[h]
            else:
                eval_result = get_or_eval(mutated)
                if eval_result is None:
                    status = "compile_fail"
                    train_acc = None
                    test_acc = None
                else:
                    status = "ok"
                    train_acc, test_acc = eval_result
                    seen.add(h)
                    cand = Candidate(mutated, train_acc, test_acc)
                    offspring.append(cand)
                    if train_acc > best_train or (train_acc == best_train and test_acc > best_test):
                        best_train = train_acc
                        best_test = test_acc
                        best_by_train = cand
                        logger.info(f"New best: train={best_train:.4f} test={best_test:.4f}")

            program_history.append(history_entry(num_programs, "generation", generation, status,
                                                 train_acc, test_acc, best_train, best_test))

            if best_train >= 1.0 or num_programs >= budget:
                break

        # (mu + lambda) replacement: keep the top mu by fitness (stable sort; no tie-break).
        population = sorted(population + offspring, key=lambda c: c.train_acc,
                            reverse=True)[:population_size]
        logger.info(f"Generation {generation} end: offspring={len(offspring)}, "
                    f"programs={num_programs}/{budget}, dup={num_duplicates}, "
                    f"fails={num_compile_failures}, best_train={best_train:.4f} "
                    f"| best_test={best_test:.4f}")

        if best_train >= 1.0 or num_programs >= budget:
            break

    logger.info(f"Completed: programs={num_programs}/{budget}, dup={num_duplicates}, "
                f"fails={num_compile_failures}, best_train={best_train:.4f}, test={best_test:.4f}")

    # NB: "complete" even when a perfect-fitness program was found in the main loop.
    return {
        "status": "complete",
        "num_programs": num_programs,
        "num_duplicates": num_duplicates,
        "num_compile_failures": num_compile_failures,
        "train_acc": best_train,
        "test_acc": best_test,
        "best_source": best_by_train.source,
        "program_history": program_history,
    }


def run_gp_for_mutant(
    mutant,
    data_dir: Path = DATA_DIR,
    budget: int = DEFAULTS["budget"],
    population_size: int = DEFAULTS["population_size"],
    tournament_k: int = DEFAULTS["tournament_k"],
    seed: int = DEFAULTS["seed"],
    log_every: int = 25,
    eval_timeout: float = 30.0,
) -> Dict[str, Any]:
    """Run GP on a `rasping.raspbugs.Mutant`; returns the per-mutant result record."""
    start_time = time.time()
    result = run_gp(
        base_source=mutant.source,
        program=mutant.program,
        data_dir=data_dir,
        budget=budget,
        population_size=population_size,
        tournament_k=tournament_k,
        seed=seed,
        log_every=log_every,
        eval_timeout=eval_timeout,
    )
    end_time = time.time()
    result.update(
        {
            # v1 keys
            "job_id": mutant.job_id,
            "program_name": mutant.program,
            "duration_s": end_time - start_time,
            "seed": seed,
            # added in the port
            "method": "gp",
            "budget": budget,
            "program": programs.canonical(mutant.program),
            "population_size": population_size,
            "tournament_k": tournament_k,
            "eval_timeout": eval_timeout,
        }
    )
    return result
