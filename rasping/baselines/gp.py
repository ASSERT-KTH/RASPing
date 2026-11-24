"""Genetic-programming (GP) repair baseline.

A (mu + lambda) evolutionary search over RASP source programs. The population is seeded
with the buggy mutant and random single-step mutations of it; each generation produces
up to `population_size` offspring by tournament selection + one random mutation, and
the best `population_size` programs (by fitness) survive. Every generated program counts
toward `budget`, including duplicates. See `common` for fitness and evaluation.

v1 (paper) settings: budget 1000, population 16, tournament 3, seed 42.

Changes from the v1 implementation (docs/FINDINGS.md): fitness on the training split
only (v1: train+val, F5); test accuracy never breaks ties (F29); build/eval failures and
timeouts are counted and excluded instead of scored 0 (F30); candidates are built with
`MAX_LEN` like the compiled models (F11); per-step timestamps (F18); consistent status
and generation numbering (F31). Pinned by `tests/test_golden_search.py`.
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
    CandidateError,
    apply_mutation_once,
    evaluate_source,
    hash_source,
    history_entry,
    load_splits,
)
from rasping.config import DATA_DIR, MAX_LEN

logger = logging.getLogger(__name__)

#: Hyperparameters of the v1 (TMLR) runs.
DEFAULTS = dict(budget=1000, population_size=16, tournament_k=3, seed=42)


@dataclass
class Candidate:
    source: str
    train_acc: float  # fitness (training-split accuracy)
    test_acc: float


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
    """Run GP from `base_source` (a mutant of `program`).

    Fitness is accuracy on the training split; test accuracy is computed for reporting
    only and never influences the search or the selected program.
    """
    rng = random.Random(seed)
    program = programs.canonical(program)
    train_data, _val, test_data = load_splits(program, data_dir)
    start = time.time()

    seen: set[str] = set()
    cache: Dict[str, Tuple[float, float]] = {}
    num_programs = num_duplicates = num_compile_failures = num_timeouts = 0
    program_history: List[Dict[str, Any]] = []
    best: Optional[Candidate] = None  # best by fitness; ties keep the incumbent

    def evaluate(src: str) -> Optional[Tuple[float, float]]:
        nonlocal num_compile_failures, num_timeouts
        h = hash_source(src)
        if h in cache:
            return cache[h]
        try:
            fit = evaluate_source(src, program, MAX_LEN, train_data, timeout=eval_timeout)
            test = evaluate_source(src, program, MAX_LEN, test_data, timeout=eval_timeout)
        except CandidateError as e:
            if e.kind == "timeout":
                num_timeouts += 1
            else:
                num_compile_failures += 1
            return None
        cache[h] = (fit, test)
        return cache[h]

    def consider(src: str, generation: int) -> Optional[Candidate]:
        """Count, evaluate and log one generated program; return it if it is new and valid."""
        nonlocal num_programs, num_duplicates, best
        num_programs += 1
        h = hash_source(src)
        if h in seen:
            num_duplicates += 1
            status, res, cand = "duplicate", cache.get(h), None
        else:
            res = evaluate(src)
            status = "compile_fail" if res is None else "ok"
            cand = None
            if res is not None:
                seen.add(h)
                cand = Candidate(src, res[0], res[1])
                if best is None or cand.train_acc > best.train_acc:
                    best = cand
        program_history.append(history_entry(
            num_programs, "generation", generation, status,
            None if res is None else res[0], None if res is None else res[1],
            -1.0 if best is None else best.train_acc, -1.0 if best is None else best.test_acc,
            time.time() - start))
        if log_every and num_programs % max(1, log_every) == 0:
            logger.info(f"programs={num_programs}/{budget} dup={num_duplicates} "
                        f"fails={num_compile_failures} timeouts={num_timeouts} "
                        f"best_fitness={-1 if best is None else best.train_acc:.4f}")
        return cand

    def done() -> bool:
        return (best is not None and best.train_acc >= 1.0) or num_programs >= budget

    # Generation 0: the buggy program plus random single-step mutations of it.
    population: List[Candidate] = []
    base = consider(base_source, 0)
    if base is not None:
        population.append(base)
    attempts = 0
    while len(population) < population_size and attempts < population_size * 20 and not done():
        attempts += 1
        mutated = apply_mutation_once(base_source, rng)
        if mutated is not None and (cand := consider(mutated, 0)) is not None:
            population.append(cand)

    if not population:
        return {"status": "failed_to_initialize", "num_programs": num_programs,
                "num_compile_failures": num_compile_failures, "num_timeouts": num_timeouts,
                "program_history": program_history}

    def tournament_select(pop: List[Candidate]) -> Candidate:
        contenders = rng.sample(pop, k=min(tournament_k, len(pop)))
        return max(contenders, key=lambda c: c.train_acc)

    generation = 0
    while not done():
        generation += 1
        offspring: List[Candidate] = []
        attempts = 0
        while len(offspring) < population_size and attempts < population_size * 50 and not done():
            attempts += 1
            mutated = apply_mutation_once(tournament_select(population).source, rng)
            if mutated is not None and (cand := consider(mutated, generation)) is not None:
                offspring.append(cand)
        # (mu + lambda) replacement by fitness (stable: earlier programs win ties)
        population = sorted(population + offspring, key=lambda c: c.train_acc,
                            reverse=True)[:population_size]

    return {
        "status": "success_early" if best.train_acc >= 1.0 else "complete",
        "num_programs": num_programs,
        "num_duplicates": num_duplicates,
        "num_compile_failures": num_compile_failures,
        "num_timeouts": num_timeouts,
        "train_acc": best.train_acc,
        "test_acc": best.test_acc,
        "best_source": best.source,
        "program_history": program_history,
        "duration_s": time.time() - start,
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
    duration_s = result.pop("duration_s", None)
    result.update(
        {
            # v1 keys
            "job_id": mutant.job_id,
            "program_name": mutant.program,
            "duration_s": duration_s,
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
