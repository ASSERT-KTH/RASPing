"""Breadth-first (exhaustive) search repair baseline.

Evaluates the buggy mutant, then every single-step mutation of it (depth 1), then every
single-step mutation of the successfully evaluated depth-1 programs (depth 2), ... until
`max_depth`, `budget` evaluated programs (duplicates included) or a perfect-fitness
program. By default there is no depth limit (v1 used the mutant's true mutation order,
an oracle; F3). Other changes from v1 as for GP: training-split fitness (F5), no test
tie-break (F29), failures counted (F30), `MAX_LEN` (F11), timestamps (F18).
Pinned by `tests/test_golden_search.py`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rasping import programs
from rasping.baselines.common import (
    CandidateError,
    all_single_step_neighbors,
    evaluate_source,
    hash_source,
    history_entry,
    load_splits,
)
from rasping.config import DATA_DIR, MAX_LEN

logger = logging.getLogger(__name__)

#: Budget of the v1 (TMLR) runs.
DEFAULTS = dict(budget=1000)


def run_bfs(
    base_source: str,
    program: str,
    data_dir: Path = DATA_DIR,
    max_depth: int | None = None,
    budget: int = DEFAULTS["budget"],
    log_every: int = 50,
    eval_timeout: float = 30.0,
) -> Dict[str, Any]:
    """Run BFS from `base_source` (a mutant of `program`) until `budget` programs.

    `max_depth=None` (default) puts no limit on the depth (v1 used the mutant's true
    mutation order; F3). Fitness is training-split accuracy; test accuracy is reported
    only. Ties keep the first program found.
    """
    program = programs.canonical(program)
    train_data, _val, test_data = load_splits(program, data_dir)
    start = time.time()

    cache: Dict[str, Tuple[float, float]] = {}
    num_programs = num_duplicates = num_compile_failures = num_timeouts = num_killed = 0
    best_train, best_test, best_source = -1.0, -1.0, base_source
    program_history: List[Dict[str, Any]] = []

    def evaluate(src: str) -> Tuple[str, Optional[Tuple[float, float]]]:
        nonlocal num_programs, num_duplicates, num_compile_failures, num_timeouts, num_killed
        num_programs += 1  # every attempt counts toward the budget
        h = hash_source(src)
        if h in cache:
            num_duplicates += 1
            return "duplicate", cache[h]
        try:
            fit = evaluate_source(src, program, MAX_LEN, train_data, timeout=eval_timeout)
            test = evaluate_source(src, program, MAX_LEN, test_data, timeout=eval_timeout)
        except CandidateError as e:
            if e.kind == "timeout":
                num_timeouts += 1
            elif e.kind == "killed":
                num_killed += 1
            else:
                num_compile_failures += 1
            return "compile_fail", None
        cache[h] = (fit, test)
        return "ok", cache[h]

    def log(status, res, depth):
        program_history.append(history_entry(
            num_programs, "depth", depth, status,
            None if res is None else res[0], None if res is None else res[1],
            best_train, best_test, time.time() - start))

    status, res = evaluate(base_source)
    if res is not None:
        best_train, best_test = res
    log(status, res, 0)

    depth, frontier = 0, [base_source]
    while best_train < 1.0 and frontier and num_programs < budget \
            and (max_depth is None or depth < max_depth):
        depth += 1
        candidates = [n for src in frontier for n in all_single_step_neighbors(src)]
        frontier = []
        for src in candidates:
            if num_programs >= budget or best_train >= 1.0:
                break
            status, res = evaluate(src)
            if status == "ok":
                frontier.append(src)
                if res[0] > best_train:
                    best_train, best_test, best_source = res[0], res[1], src
            log(status, res, depth)
            if log_every and num_programs % max(1, log_every) == 0:
                logger.info(f"depth={depth} programs={num_programs}/{budget} dup={num_duplicates} "
                            f"fails={num_compile_failures} timeouts={num_timeouts} "
                            f"best_fitness={best_train:.4f}")

    return {
        "status": "success_early" if best_train >= 1.0 else "complete",
        "num_programs": num_programs,
        "num_duplicates": num_duplicates,
        "num_compile_failures": num_compile_failures,
        "num_timeouts": num_timeouts,
        "num_killed": num_killed,
        "train_acc": best_train,
        "test_acc": best_test,
        "best_source": best_source,
        "program_history": program_history,
        "duration_s": time.time() - start,
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

    `max_depth=None` means unlimited (budget only). To reproduce v1, pass
    `max_depth=mutant.mutation_order` (F3).
    """
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
