"""Submit one symbolic-baseline job (GP or BFS) per mutant of a benchmark (CPU jobs).

Each job runs `experiments/symbolic_baselines/run.py` and writes
`runs/symbolic_baselines/<method>_budget<B>/<job_id>.json`; mutants whose record already
exists are skipped, so re-running the launcher resubmits only missing jobs. Collect the
records into tables with `experiments/symbolic_baselines/collect.py`.

    python experiments/symbolic_baselines/launch.py --method gp     # v2: budget 5,000
    python experiments/symbolic_baselines/launch.py --method bfs

Budget calibration on the init pilot's sample (25 per program, stratified by order):

    python experiments/symbolic_baselines/launch.py --method bfs --budget 5000 \
        --n-per-program 25 --timeout-hours 24 --partition <cpu partition>

On the 847 benchmark mutants, v1 at budget 1,000 took at most 3.5 h per run (median
0.7 h GP, 0.9 h BFS) and 664 / 891 CPU-hours in total. A run's record is written only when
it finishes, so set --timeout-hours above the longest expected run.

Memory: candidates are evaluated in forked workers, and a worker the OOM killer stops is
counted as `num_killed`, not as a failed candidate (F37). The launcher requests --mem-gb
(32 GB). In the budget-5000 calibration, 199 of 200 jobs peaked below 4.1 GB, but one GP job
had 8 workers killed under the 1-CPU default of 7.9 GB, and its rerun with 32 GB peaked at
18.5 GB. A record with num_killed > 0 should be rerun with more memory.

The v1 (TMLR) runs in `results/v1_tmlr/symbolic_baselines/` covered the 1,466 v1 mutants
(Dyck programs included), with BFS limited to depth = mutation order (F3, since fixed).
The budget-500 tables in `results/v1_tmlr/spec_size/` came from an earlier round over a
different, unrecorded set of mutants (1,691 GP / 1,760 BFS records).
"""

from __future__ import annotations

import sys

import click

from rasping import cluster
from rasping.baselines import METHODS, experiment, run_dir
from rasping.benchmarks import BENCHMARKS
from rasping.config import BASELINE_BUDGET

RUN_SCRIPT = "experiments/symbolic_baselines/run.py"


@click.command()
@click.option("--method", type=click.Choice(METHODS), required=True)
@click.option("--benchmark", type=click.Choice(sorted(BENCHMARKS)), default="raspbugs", show_default=True)
@click.option("--budget", type=int, default=BASELINE_BUDGET, show_default=True)
@click.option("--n-per-program", type=int, default=None,
              help="Only a stratified sample of this many mutants per program (Benchmark.stratified_sample)")
@click.option("--sample-seed", type=int, default=0, show_default=True)
@click.option("--timeout-hours", type=float, default=12, show_default=True, help="SLURM time limit per job")
@click.option("--partition", default=None, help="SLURM partition (e.g. the cluster's CPU partition)")
@click.option("--mem-gb", type=int, default=32, show_default=True, help="SLURM memory per job (F37)")
@cluster.add_cli_options
def main(method, benchmark, budget, n_per_program, sample_seed, timeout_hours, partition, mem_gb,
         dry_run, local, no_container, limit):
    bench = BENCHMARKS[benchmark]
    out = run_dir(method, budget, benchmark=benchmark)
    python = sys.executable if no_container else "python"
    if n_per_program is None:
        mutants = bench.mutants()
    else:
        mutants = [bench.get(j) for p in bench.programs()
                   for j in bench.stratified_sample(p, n_per_program, sample_seed)]
    todo = [m for m in mutants if not (out / f"{m.job_id}.json").exists()]
    print(f"{benchmark} {method} budget {budget}: {len(mutants) - len(todo)}/{len(mutants)} done, "
          f"{len(todo)} to run")
    commands = [[python, RUN_SCRIPT, "--method", method, "--job-id", m.job_id,
                 "--budget", str(budget)] for m in todo]
    cluster.submit_from_cli(
        commands, f"{experiment(benchmark)}_{method}_budget{budget}", dry_run, local, no_container, limit,
        gpu=False, timeout_min=int(timeout_hours * 60), mem_gb=mem_gb,
        **({"slurm_partition": partition} if partition else {}),
        slurm_additional_parameters={},  # CPU-only: drop the default GPU (MIG) reservation
    )


if __name__ == "__main__":
    main()
