"""Specification-size ablation (paper §5.4, App. D): repair with N training examples.

For each mutant and N, trains on a seeded subsample of N training examples (and N//8
validation examples) with a fixed budget of 500k gradient steps:

    python experiments/spec_size/launch.py --dry-run
    python -m rasping.results collect spec_size --version v2

v1 (TMLR): N in {100, 1000, 5000, 10000, 25000}, seed 42, the v1 mutants of every program
but shuffle_dyck (the analysis kept the four main programs); the
N=40,000 column of Table 8 reused the main runs, which had no 500k-step cap (F8). Now
N=40,000 is run within the ablation under the same budget.
"""

import click

from rasping import cluster, raspbugs
from rasping.config import RUNS_DIR

N_VALUES = (100, 1000, 5000, 10000, 25000, 40000)
N_STEPS = 500_000


@click.command()
@click.option("--n", "n_values", type=int, multiple=True, default=N_VALUES, show_default=True)
@click.option("--seed", "seeds", type=int, multiple=True, default=(42,), show_default=True)
@click.option("--n-steps", type=int, default=N_STEPS, show_default=True)
@cluster.add_cli_options
def main(n_values, seeds, n_steps, **cli):
    root = RUNS_DIR / "spec_size"
    commands = []
    for m in raspbugs.mutants():
        for n in n_values:
            for seed in seeds:
                out = root / m.program / f"n_{n}" / f"seed_{seed}" / m.job_id
                if (out / "run.json").exists():
                    continue
                commands.append(["python", "-m", "rasping.repair", "--job-id", m.job_id,
                                 "--out", str(out), "--n-train", str(n), "--n-val", str(n // 8),
                                 "--seed", str(seed), "--n-steps", str(n_steps), "--no-weights"])
    cluster.submit_from_cli(commands, "spec_size", **cli)


if __name__ == "__main__":
    main()
