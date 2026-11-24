"""Submit one symbolic-baseline job (GP or BFS) per RaspBugs mutant (CPU jobs).

Each job runs `experiments/symbolic_baselines/run.py` and writes
`runs/symbolic_baselines/<method>_budget<B>/<job_id>.json`; mutants whose record already
exists are skipped, so re-running the launcher resubmits only missing jobs. Collect the
records into tables with `experiments/symbolic_baselines/collect.py`.

    python experiments/symbolic_baselines/launch.py --method gp --budget 1000
    python experiments/symbolic_baselines/launch.py --method bfs --budget 1000

The v1 (TMLR) runs in `results/v1_tmlr/symbolic_baselines/` covered the 1,466 v1 mutants
(Dyck programs included), with BFS limited to depth = mutation order (F3, since fixed).
The budget-500 tables in `results/v1_tmlr/spec_size/` came from an earlier round over a
different, unrecorded set of mutants (1,691 GP / 1,760 BFS records).
"""

from __future__ import annotations

import sys

import click

from rasping import cluster, raspbugs
from rasping.baselines import METHODS, run_dir

RUN_SCRIPT = "experiments/symbolic_baselines/run.py"


@click.command()
@click.option("--method", type=click.Choice(METHODS), required=True)
@click.option("--budget", type=int, default=1000, show_default=True)
@cluster.add_cli_options
def main(method, budget, dry_run, local, no_container, limit):
    out = run_dir(method, budget)
    python = sys.executable if no_container else "python"
    mutants = raspbugs.mutants()
    todo = [m for m in mutants if not (out / f"{m.job_id}.json").exists()]
    print(f"{method} budget {budget}: {len(mutants) - len(todo)}/{len(mutants)} done, "
          f"{len(todo)} to run")
    commands = [[python, RUN_SCRIPT, "--method", method, "--job-id", m.job_id,
                 "--budget", str(budget)] for m in todo]
    cluster.submit_from_cli(
        commands, f"symbolic_{method}_budget{budget}", dry_run, local, no_container, limit,
        gpu=False,
        slurm_additional_parameters={},  # CPU-only: drop the default GPU (MIG) reservation
    )


if __name__ == "__main__":
    main()
