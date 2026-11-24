"""Collect symbolic-baseline result records into tidy tables (the v1 schema).

    runs/symbolic_baselines/<method>_budget<B>/*.json
        -> results/<version>/symbolic_baselines/<method>_budget<B>_runs.csv
           results/<version>/symbolic_baselines/<method>_budget<B>_history.csv.gz

Columns and row order match `results/v1_tmlr/symbolic_baselines/` (see
`rasping.baselines.records`, shared with `scripts/freeze_v1_results.py`).

    python experiments/symbolic_baselines/collect.py --version v2 [--method gp] [--budget 1000]
"""

from __future__ import annotations

from pathlib import Path

import click

from rasping.baselines import EXPERIMENT, METHODS, records, run_dir
from rasping.config import RUNS_DIR
from rasping.results import results_dir


@click.command()
@click.option("--version", default=None, help="Results version, e.g. v2 (-> results/<version>/)")
@click.option("--method", type=click.Choice(METHODS), default=None,
              help="Collect only this method [default: all]")
@click.option("--budget", type=int, default=1000, show_default=True)
@click.option("--runs-dir", type=click.Path(path_type=Path), default=RUNS_DIR, show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=None,
              help="Write the tables here instead of results/<version>/symbolic_baselines")
def main(version, method, budget, runs_dir, out_dir):
    if out_dir is None:
        if version is None:
            raise click.UsageError("give --version (or --out-dir)")
        out_dir = results_dir(version, EXPERIMENT)
    for m in [method] if method else METHODS:
        src = run_dir(m, budget, runs_dir)
        if not src.is_dir():
            print(f"skip {m}: no records in {src}")
            continue
        runs, hist = records.tables(records.load_records(src), m, budget)
        runs_path, hist_path = records.table_paths(out_dir, m, budget)
        records.write_csv(runs, runs_path)
        records.write_csv(hist, hist_path)
        print(f"{m}: {len(runs)} runs, {len(hist)} history rows -> {runs_path.parent}")


if __name__ == "__main__":
    main()
