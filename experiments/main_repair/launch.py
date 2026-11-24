"""Launch gradient-based repair on every RaspBugs mutant (paper §5.2; Table 1, Figs 3/7).

One SLURM job per mutant, each running `python -m rasping.repair` in the container and
writing `runs/main_repair/<program>/<job_id>/`. Already finished runs are skipped.

    python experiments/main_repair/launch.py --dry-run          # show the commands
    python experiments/main_repair/launch.py                    # submit
    python -m rasping.results collect main_repair --version v2  # then collect

v1 (TMLR) runs: all buggy mutants, default TrainConfig (10k epochs, batch 256, lr 1e-4,
patience 50), trajectories stored every 50 steps.
"""

import click

from rasping import cluster, raspbugs
from rasping.config import RUNS_DIR


@click.command()
@click.option("--subset", type=click.Choice(raspbugs.SUBSETS), default="all", show_default=True)
@click.option("--extra", default="", help="Extra arguments passed to rasping.repair")
@cluster.add_cli_options
def main(subset, extra, **cli):
    root = RUNS_DIR / "main_repair"
    commands = []
    for m in raspbugs.mutants(subset):
        out = root / m.program / m.job_id
        if (out / "run.json").exists():
            continue
        commands.append(["python", "-m", "rasping.repair", "--job-id", m.job_id, "--out", str(out),
                         *extra.split()])
    cluster.submit_from_cli(commands, "main_repair", **cli)


if __name__ == "__main__":
    main()
