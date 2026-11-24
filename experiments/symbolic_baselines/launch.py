"""Submit one symbolic-baseline job (GP or BFS) per RaspBugs mutant (CPU jobs).

Each job runs `experiments/symbolic_baselines/run.py` and writes
`runs/symbolic_baselines/<method>_budget<B>/<job_id>.json`; mutants whose record already
exists are skipped, so re-running the launcher resubmits only missing jobs. Collect the
records into tables with `experiments/symbolic_baselines/collect.py`.

    python experiments/symbolic_baselines/launch.py --method gp --budget 1000
    python experiments/symbolic_baselines/launch.py --method bfs --budget 1000

Budget calibration on the init pilot's sample (25 per program, stratified by order):

    python experiments/symbolic_baselines/launch.py --method bfs --budget 5000 \
        --n-per-program 25 --timeout-hours 24 --partition <cpu partition>

On the 847 benchmark mutants, v1 at budget 1,000 took at most 3.5 h per run (median
0.7 h GP, 0.9 h BFS) and 664 / 891 CPU-hours in total. A run's record is written only when
it finishes, so set --timeout-hours above the longest expected run.

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
from rasping.config import PROGRAMS

RUN_SCRIPT = "experiments/symbolic_baselines/run.py"


@click.command()
@click.option("--method", type=click.Choice(METHODS), required=True)
@click.option("--budget", type=int, default=1000, show_default=True)
@click.option("--n-per-program", type=int, default=None,
              help="Only a stratified sample of this many mutants per program (raspbugs.stratified_sample)")
@click.option("--sample-seed", type=int, default=0, show_default=True)
@click.option("--timeout-hours", type=float, default=12, show_default=True, help="SLURM time limit per job")
@click.option("--partition", default=None, help="SLURM partition (e.g. the cluster's CPU partition)")
@cluster.add_cli_options
def main(method, budget, n_per_program, sample_seed, timeout_hours, partition, dry_run, local,
         no_container, limit):
    out = run_dir(method, budget)
    python = sys.executable if no_container else "python"
    if n_per_program is None:
        mutants = raspbugs.mutants()
    else:
        mutants = [raspbugs.get(j) for p in PROGRAMS
                   for j in raspbugs.stratified_sample(p, n_per_program, sample_seed)]
    todo = [m for m in mutants if not (out / f"{m.job_id}.json").exists()]
    print(f"{method} budget {budget}: {len(mutants) - len(todo)}/{len(mutants)} done, "
          f"{len(todo)} to run")
    commands = [[python, RUN_SCRIPT, "--method", method, "--job-id", m.job_id,
                 "--budget", str(budget)] for m in todo]
    cluster.submit_from_cli(
        commands, f"symbolic_{method}_budget{budget}", dry_run, local, no_container, limit,
        gpu=False, timeout_min=int(timeout_hours * 60),
        **({"slurm_partition": partition} if partition else {}),
        slurm_additional_parameters={},  # CPU-only: drop the default GPU (MIG) reservation
    )


if __name__ == "__main__":
    main()
