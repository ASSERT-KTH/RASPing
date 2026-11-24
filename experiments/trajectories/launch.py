"""Re-run the repair of the mutants shown in Figures 1 and 6, storing trajectories.

    python experiments/trajectories/launch.py
    python experiments/trajectories/plot.py plot-file \
        --trajectory-file runs/trajectories/sort/92e80c60de964d53b02dc7e968a0559e/trajectory.pkl \
        --output-dir paper_artifacts/<version>/figures/trajectories
"""

import click

from rasping import cluster, raspbugs
from rasping.config import RUNS_DIR

#: (figure, job id) of the trajectories shown in the v1 paper.
FIGURE_JOBS = {
    "fig1": "231a32a5efd24a10840857bd72e478a1",  # reverse
    "fig6": "92e80c60de964d53b02dc7e968a0559e",  # sort
}


@click.command()
@click.option("--interval", default=50, show_default=True, help="Store parameters every N steps")
@cluster.add_cli_options
def main(interval, **cli):
    commands = []
    for job_id in FIGURE_JOBS.values():
        out = RUNS_DIR / "trajectories" / raspbugs.get(job_id).program / job_id
        commands.append(["python", "-m", "rasping.repair", "--job-id", job_id, "--out", str(out),
                         "--store-trajectory", "--trajectory-interval", str(interval)])
    cluster.submit_from_cli(commands, "trajectories", **cli)


if __name__ == "__main__":
    main()
