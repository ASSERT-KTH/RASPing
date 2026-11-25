"""Submit TracrBench mutation generation (step 1) as one CPU job per program.

Each job runs `raspbugs/run_mutations.py --root tracrbench --program <name>` (orders 1-5 in
sequence, in a temporary copy of the program) and writes tracrbench/results/<name>/.
Programs whose results already hold all five orders are skipped.

    python tracrbench/launch_mutations.py --dry-run
    python tracrbench/launch_mutations.py --partition <cpu partition>
    python raspbugs/build_manifest.py --root tracrbench          # step 2, afterwards
"""

from __future__ import annotations

import sys

import click

from rasping import cluster, programs
from rasping.config import TRACRBENCH_DIR

ORDERS = (1, 2, 3, 4, 5)


@click.command()
@click.option("--program", "only", multiple=True, help="Only these programs")
@click.option("--timeout-hours", type=float, default=24, show_default=True, help="SLURM time limit per job")
@click.option("--partition", default=None, help="SLURM partition (the cluster's CPU partition)")
@click.option("--mem-gb", type=int, default=16, show_default=True)
@cluster.add_cli_options
def main(only, timeout_hours, partition, mem_gb, dry_run, local, no_container, limit):
    python = sys.executable if no_container else "python"
    names = list(only) or [p.name for p in programs.tracrbench_programs()]
    done = [n for n in names if all(
        (TRACRBENCH_DIR / "results" / n / f"tracr_{n}_order={o}.jsonl").exists() for o in ORDERS)]
    todo = [n for n in names if n not in done]
    print(f"{len(done)}/{len(names)} programs done, {len(todo)} to run")
    commands = [[python, "raspbugs/run_mutations.py", "--root", "tracrbench", "--program", n,
                 "--orders", ",".join(map(str, ORDERS)), "--workers", "1"] for n in todo]
    cluster.submit_from_cli(
        commands, "tracrbench_mutations", dry_run, local, no_container, limit,
        gpu=False, timeout_min=int(timeout_hours * 60), mem_gb=mem_gb,
        **({"slurm_partition": partition} if partition else {}),
        slurm_additional_parameters={},  # CPU-only: drop the default GPU (MIG) reservation
    )


if __name__ == "__main__":
    main()
