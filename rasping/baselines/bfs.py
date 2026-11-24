"""Breadth-first (exhaustive) search repair baseline.

Evaluates the buggy mutant, then every single-step mutation of it (depth 1), then every
single-step mutation of the successfully evaluated depth-1 programs (depth 2), ... until
`max_depth`, `budget` evaluated programs (duplicates included) or a perfect-fitness
program. The v1 runs set `max_depth` to the mutant's true mutation order (F3); that is
what `run_bfs_for_mutant` does by default.

Ported from `experiments/gp_baseline/run_gp_exhaustive.py` with identical behaviour; it
is pinned by `tests/test_golden_search.py`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rasping import programs
from rasping.baselines.common import (
    all_single_step_neighbors,
    compute_max_length,
    evaluate_source,
    hash_source,
    history_entry,
    load_splits,
)
from rasping.config import DATA_DIR

logger = logging.getLogger(__name__)

#: Budget of the v1 (TMLR) runs (max_depth = the mutant's mutation order).
DEFAULTS = dict(budget=1000)


def run_bfs(
    base_source: str,
    program: str,
    data_dir: Path = DATA_DIR,
    max_depth: int = 2,
    budget: int = DEFAULTS["budget"],
    log_every: int = 50,
    eval_timeout: float = 30.0,
) -> Dict[str, Any]:
    """Run BFS from `base_source` (a mutant of `program`). Returns the v1 result dict
    with `duration_s` (measured from the base evaluation, i.e. excluding data loading)
    but without the job metadata that `run_bfs_for_mutant` adds."""
    program = programs.canonical(program)

    train_data, val_data, test_data = load_splits(program, data_dir)
    fitness_data = train_data + val_data  # F5: fitness uses train+val
    max_length = compute_max_length(program, data_dir)

    cache: Dict[str, Tuple[float, float]] = {}
    num_programs = 0
    num_duplicates = 0
    num_compile_failures = 0

    best_train = -1.0
    best_test = -1.0
    best_source = base_source

    program_history: List[Dict[str, Any]] = []

    def evaluate_candidate(src: str) -> Tuple[str, Optional[Tuple[float, float]], str]:
        """Evaluate a candidate; returns (status, (train, test) or None, hash)."""
        nonlocal num_programs, num_compile_failures, num_duplicates
        h = hash_source(src)
        num_programs += 1  # every attempt counts toward the budget
        if h in cache:
            num_duplicates += 1
            return "duplicate", cache[h], h
        try:
            train_acc = evaluate_source(src, program, max_length, fitness_data, timeout=eval_timeout)
            test_acc = evaluate_source(src, program, max_length, test_data, timeout=eval_timeout)
        except Exception as e:  # unreachable: evaluate_source never raises
            num_compile_failures += 1
            logger.debug(f"Compile/eval failed for candidate {h}: {e}")
            return "compile_fail", None, h
        cache[h] = (train_acc, test_acc)
        return "ok", (train_acc, test_acc), h

    start_time = time.time()
    status, base_eval, _ = evaluate_candidate(base_source)
    if status != "compile_fail" and base_eval is not None:
        best_train, best_test = base_eval
        logger.info(f"Seeded base candidate: train={best_train:.4f} test={best_test:.4f}")
    program_history.append(history_entry(
        num_programs, "depth", 0, status,
        None if base_eval is None else base_eval[0],
        None if base_eval is None else base_eval[1],
        best_train, best_test))

    depth = 0
    frontier: List[str] = [base_source]

    if best_train < 1.0:
        while depth < max_depth and frontier and num_programs < budget:
            # Expand every program of the frontier before evaluating anything.
            next_frontier: List[str] = []
            for src in frontier:
                next_frontier.extend(all_single_step_neighbors(src))

            depth += 1
            new_frontier: List[str] = []
            for src in next_frontier:
                if num_programs >= budget:
                    break
                status_i, eval_res, _ = evaluate_candidate(src)
                train_acc = eval_res[0] if eval_res is not None else None
                test_acc = eval_res[1] if eval_res is not None else None
                if status_i == "ok" and eval_res is not None:
                    new_frontier.append(src)
                    # NB: ties on fitness are broken by *test* accuracy
                    if train_acc > best_train or (train_acc == best_train and test_acc > best_test):
                        best_train = train_acc
                        best_test = test_acc
                        best_source = src
                        logger.info(f"New best: train={best_train:.4f} test={best_test:.4f}")
                program_history.append(history_entry(num_programs, "depth", depth, status_i,
                                                     train_acc, test_acc, best_train, best_test))
                if log_every and num_programs % max(1, log_every) == 0:
                    logger.info(f"Progress: depth={depth} programs={num_programs}/{budget} "
                                f"dup={num_duplicates} fails={num_compile_failures} "
                                f"best_train={best_train:.4f} best_test={best_test:.4f}")
                if best_train >= 1.0:
                    break

            frontier = new_frontier

            if best_train >= 1.0 or num_programs >= budget:
                break

    end_time = time.time()
    return {
        "status": "success_early" if best_train >= 1.0 else "complete",
        "num_programs": num_programs,
        "num_duplicates": num_duplicates,
        "num_compile_failures": num_compile_failures,
        "train_acc": best_train,
        "test_acc": best_test,
        "best_source": best_source,
        "program_history": program_history,
        "duration_s": end_time - start_time,
    }


def run_bfs_for_mutant(
    mutant,
    data_dir: Path = DATA_DIR,
    max_depth: int | None = None,
    budget: int = DEFAULTS["budget"],
    log_every: int = 50,
    eval_timeout: float = 30.0,
) -> Dict[str, Any]:
    """Run BFS on a `rasping.raspbugs.Mutant`; returns the per-mutant result record.

    `max_depth=None` uses the mutant's mutation order, as the v1 runs did (F3).
    """
    if max_depth is None:
        max_depth = mutant.mutation_order
    result = run_bfs(
        base_source=mutant.source,
        program=mutant.program,
        data_dir=data_dir,
        max_depth=max_depth,
        budget=budget,
        log_every=log_every,
        eval_timeout=eval_timeout,
    )
    duration_s = result.pop("duration_s")
    result.update(
        {
            # v1 keys
            "job_id": mutant.job_id,
            "program_name": mutant.program,
            "duration_s": duration_s,
            # added in the port
            "method": "bfs",
            "budget": budget,
            "program": programs.canonical(mutant.program),
            "max_depth": max_depth,
            "eval_timeout": eval_timeout,
        }
    )
    return result
