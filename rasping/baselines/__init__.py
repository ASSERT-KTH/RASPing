"""Symbolic repair baselines: genetic programming (`gp`) and breadth-first search (`bfs`)
over RASP source programs, using the benchmark's mutation operators (paper §5.3).

    from rasping.benchmarks import get_mutant
    from rasping.baselines import gp, bfs
    rec = gp.run_gp_for_mutant(get_mutant(job_id), budget=1000)

Per-mutant result records live in `runs/symbolic_baselines/<method>_budget<B>/<job_id>.json`
for RaspBugs and `runs/symbolic_baselines_tracrbench/...` for TracrBench (see
`experiments/symbolic_baselines/`); `records` turns them into tidy tables.
"""

from __future__ import annotations

from pathlib import Path

from rasping.config import RUNS_DIR

METHODS = ("gp", "bfs")
EXPERIMENT = "symbolic_baselines"


def experiment(benchmark: str = "raspbugs") -> str:
    """Directory name under runs/ and results/<version>/ of a benchmark's baseline runs."""
    return EXPERIMENT if benchmark == "raspbugs" else f"{EXPERIMENT}_{benchmark}"


def run_dir(method: str, budget: int, runs_dir: str | Path = RUNS_DIR,
            benchmark: str = "raspbugs") -> Path:
    """Directory holding the per-mutant result records of one (method, budget)."""
    return Path(runs_dir) / experiment(benchmark) / f"{method}_budget{budget}"
