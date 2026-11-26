"""Collect raw run directories into tidy result tables, and load them back.

Raw runs (gitignored) live under `runs/<experiment>/...`; each has a `run.json` and
`curves.npz` written by `rasping.repair`. `collect()` turns a tree of runs into
    results/<version>/<experiment>/runs.csv        one row per run
    results/<version>/<experiment>/curves.csv.gz   one row per (run, epoch)
    results/<version>/<experiment>/val_curves.csv.gz one row per (run, validation eval)
which is all the analysis scripts read. `results/v1_tmlr/` was produced from the
legacy layout by a one-off script with the same core columns (`scripts/freeze_v1_results.py`,
removed after commit 992e2c0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rasping.benchmarks import BENCHMARKS
from rasping.config import RESULTS_DIR

#: Columns that identify a run (besides job_id) in each experiment's runs.csv.
RUN_KEYS = ["job_id", "init", "seed", "n_train"]


def results_dir(version: str, experiment: str | None = None) -> Path:
    d = RESULTS_DIR / version
    return d / experiment if experiment else d


def collect(run_root: str | Path, out_dir: str | Path, with_curves: bool = True) -> pd.DataFrame:
    """Collect every `run.json` under `run_root` into tidy tables in `out_dir`."""
    rows, curves, val_curves = [], [], []
    for rj in sorted(Path(run_root).rglob("run.json")):
        rec = json.loads(rj.read_text())
        cfg = rec.pop("config", {})
        times = rec.pop("epoch_end_times_s", [])
        row = {**rec, **{f"cfg_{k}": v for k, v in cfg.items()}, "run_dir": str(rj.parent)}
        npz = rj.parent / "curves.npz"
        if npz.exists():
            c = np.load(npz)
            row["n_val_evals"] = len(c["val_accs"]) if "val_accs" in c else 0
            row["final_val_acc"] = float(c["val_accs"][-1]) if row["n_val_evals"] else np.nan
            if with_curves:
                key = {k: rec[k] for k in RUN_KEYS}
                for e in range(len(c["train_losses"])):
                    curves.append({**key, "epoch": e,
                                   "train_loss": float(c["train_losses"][e]),
                                   "train_acc": float(c["train_accs"][e]),
                                   "val_loss": float(c["val_losses"][e]) if e < len(c["val_losses"]) else np.nan,
                                   "time_s": times[e] if e < len(times) else np.nan})
                sel = c["selected_test_accs"] if "selected_test_accs" in c else []
                for i, v in enumerate(c["val_accs"]):
                    val_curves.append({**key, "val_eval": i, "val_acc": float(v),
                                       "selected_test_acc": float(sel[i]) if i < len(sel) else np.nan})
        rows.append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False, float_format="%.10g")
    if with_curves:
        pd.DataFrame(curves).to_csv(out / "curves.csv.gz", index=False, float_format="%.10g")
        pd.DataFrame(val_curves).to_csv(out / "val_curves.csv.gz", index=False, float_format="%.10g")
    print(f"collected {len(runs)} runs from {run_root} -> {out}")
    return runs


def load_runs(version: str, experiment: str, name: str = "runs.csv",
              with_mutant_info: bool = True) -> pd.DataFrame:
    """Load a tidy table, adding program / mutation_order for each job_id (see `mutant_table`)."""
    df = pd.read_csv(results_dir(version, experiment) / name)
    if with_mutant_info and "job_id" in df:
        meta = mutant_table(version)
        drop = [c for c in ("program", "mutation_order", "benchmark") if c in df.columns]
        df = df.drop(columns=drop).merge(meta, on="job_id", how="left")
    return df


def mutant_table(version: str | None = None) -> pd.DataFrame:
    """job_id -> program, mutation_order (and benchmark).

    A results version can freeze its own mutant table as `results/<version>/mutants.csv`.
    v1_tmlr does: it records the submitted paper's 1,466-mutant benchmark, Dyck programs
    included. Otherwise the table comes from the current manifests of both benchmarks
    (RaspBugs and TracrBench; their job ids are distinct).
    """
    frozen = results_dir(version) / "mutants.csv" if version else None
    if frozen is not None and frozen.exists():
        return pd.read_csv(frozen, usecols=["job_id", "program", "mutation_order"])
    ms = [m for b in BENCHMARKS.values() for m in b.mutants()]
    return pd.DataFrame({"job_id": [m.job_id for m in ms], "program": [m.program for m in ms],
                         "mutation_order": [m.mutation_order for m in ms],
                         "benchmark": [m.benchmark for m in ms]})


#: Experiments whose runs are written by `rasping.repair` (collected with `collect`).
REPAIR_EXPERIMENTS = ("main_repair", "tracrbench_repair", "spec_size", "init_control", "init_pilot",
                      "trajectories")


def _cli():
    import click

    from rasping.config import RUNS_DIR

    @click.group()
    def cli():
        """Collect raw runs into tidy result tables."""

    @cli.command("collect")
    @click.argument("experiment", type=click.Choice(REPAIR_EXPERIMENTS))
    @click.option("--version", required=True, help="Results version to write, e.g. v2")
    @click.option("--runs", default=None, help=f"Run root (default {RUNS_DIR}/<experiment>)")
    @click.option("--no-curves", is_flag=True)
    def collect_cmd(experiment, version, runs, no_curves):
        collect(runs or RUNS_DIR / experiment, results_dir(version, experiment), not no_curves)
        sel = Path(runs or RUNS_DIR / experiment) / "selected_jobs.json"
        if sel.exists():
            (results_dir(version, experiment) / "selected_jobs.json").write_text(sel.read_text())

    cli()


if __name__ == "__main__":
    _cli()
