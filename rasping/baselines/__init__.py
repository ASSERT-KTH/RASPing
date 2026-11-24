"""Symbolic repair baselines: genetic programming (`gp`) and breadth-first search (`bfs`)
over RASP source programs, using the benchmark's mutation operators (paper §5.3).

    from rasping import raspbugs
    from rasping.baselines import gp, bfs
    rec = gp.run_gp_for_mutant(raspbugs.get(job_id), budget=1000)

Per-mutant result records live in `runs/symbolic_baselines/<method>_budget<B>/<job_id>.json`
(see `experiments/symbolic_baselines/`); `records` turns them into tidy tables.
"""

from __future__ import annotations

from pathlib import Path

from rasping.config import RUNS_DIR

METHODS = ("gp", "bfs")
EXPERIMENT = "symbolic_baselines"


def run_dir(method: str, budget: int, runs_dir: str | Path = RUNS_DIR) -> Path:
    """Directory holding the per-mutant result records of one (method, budget)."""
    return Path(runs_dir) / EXPERIMENT / f"{method}_budget{budget}"
