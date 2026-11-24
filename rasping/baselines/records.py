"""Per-mutant baseline result records -> tidy tables (the v1 schema).

    <method>_budget<B>_runs.csv        one row per mutant
    <method>_budget<B>_history.csv.gz  sparse search history: only the evaluations at
                                       which (best_train_acc, best_test_acc) changes,
                                       plus the first and last evaluation

Used by `experiments/symbolic_baselines/collect.py` and `scripts/freeze_v1_results.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from rasping import programs


def run_row(rec: dict, method: str, budget: int) -> dict:
    h = rec.get("program_history", [])
    return {
        "method": method,
        "budget": budget,
        "job_id": rec["job_id"],
        "program_file": rec["program_name"],
        "status": rec.get("status"),
        "num_programs": rec.get("num_programs"),
        "num_duplicates": rec.get("num_duplicates"),
        "num_compile_failures": rec.get("num_compile_failures"),
        **({"num_timeouts": rec["num_timeouts"]} if "num_timeouts" in rec else {}),
        "train_acc": rec.get("train_acc"),
        "test_acc": rec.get("test_acc"),
        "max_best_test_acc": max(((e.get("best_test_acc") or 0.0) for e in h), default=0.0),
        "duration_s": rec.get("duration_s"),
        "seed": rec.get("seed"),
        "history_len": len(h),
        "max_step": max((int(e.get("step", 0)) for e in h), default=0),
    }


def history_rows(rec: dict, method: str, budget: int) -> list[dict]:
    """Change points of the best-so-far values (+ the first and last entry)."""
    h = rec.get("program_history", [])
    rows, prev = [], None
    for i, e in enumerate(h):
        key = (e.get("best_train_acc"), e.get("best_test_acc"))
        if key != prev or i == len(h) - 1:
            rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "job_id": rec["job_id"],
                    "step": e.get("step"),
                    "generation": e.get("generation", e.get("depth")),
                    "status": e.get("status"),
                    "train_acc": e.get("train_acc"),
                    "test_acc": e.get("test_acc"),
                    "best_train_acc": e.get("best_train_acc"),
                    "best_test_acc": e.get("best_test_acc"),
                    # v2 records only (wall-clock seconds since the search started)
                    **({"time_s": e["time_s"]} if "time_s" in e else {}),
                }
            )
            prev = key
    return rows


def tables(records: Iterable[dict], method: str, budget: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs, history) tables of `records` (in the given order)."""
    runs, hist = [], []
    for rec in records:
        runs.append(run_row(rec, method, budget))
        hist.extend(history_rows(rec, method, budget))
    return pd.DataFrame(runs), pd.DataFrame(hist)


def load_records(results_dir: str | Path) -> list[dict]:
    """Every result record (`*.json` with a `job_id`) in `results_dir`, sorted by
    `<data_name>_<job_id>` (the v1 file naming, hence the v1 row order)."""
    recs = []
    for fp in Path(results_dir).glob("*.json"):
        rec = json.loads(fp.read_text())
        if "job_id" in rec:
            recs.append(rec)
    return sorted(recs, key=lambda r: f"{programs.get(r['program_name']).data_name}_{r['job_id']}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


def table_paths(out_dir: str | Path, method: str, budget: int) -> tuple[Path, Path]:
    stem = Path(out_dir) / f"{method}_budget{budget}"
    return Path(f"{stem}_runs.csv"), Path(f"{stem}_history.csv.gz")
