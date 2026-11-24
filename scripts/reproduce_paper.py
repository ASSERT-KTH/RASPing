"""Regenerate every paper table and figure from the tidy result tables.

    python scripts/reproduce_paper.py                   # v1_tmlr (the TMLR submission)
    python scripts/reproduce_paper.py --version v2

Runs each experiment's analyze.py; outputs go to paper_artifacts/<version>/. No GPU and
no raw runs needed. Figures 1 and 6 (trajectories) need raw trajectory.pkl files and
are produced separately by experiments/trajectories/plot.py.
"""

import subprocess
import sys

import click

from rasping.config import ARTIFACTS_DIR, REPO_ROOT, RESULTS_DIR

#: (script, paper artifacts it produces)
ANALYSES = [
    ("experiments/main_repair/analyze.py", "Table 1, Table 5, Fig 3, Fig 7"),
    ("experiments/symbolic_baselines/analyze.py", "Fig 4, Fig 5"),
    ("experiments/spec_size/analyze.py", "Table 8, Fig 8"),
    ("experiments/init_control/analyze.py", "Table 2, Table 9, Fig 9, Fig 10"),
]


@click.command()
@click.option("--version", default="v1_tmlr", show_default=True)
def main(version):
    if not (RESULTS_DIR / version).exists():
        sys.exit(f"No results for version {version!r} in {RESULTS_DIR}")
    failed = []
    for script, what in ANALYSES:
        print(f"\n=== {script}  ({what})", flush=True)
        r = subprocess.run([sys.executable, str(REPO_ROOT / script), "--version", version])
        if r.returncode:
            failed.append(script)
    print(f"\nArtifacts in {ARTIFACTS_DIR / version}")
    if failed:
        sys.exit(f"Failed: {failed}")


if __name__ == "__main__":
    main()
