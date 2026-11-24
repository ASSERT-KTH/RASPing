"""Paired initialization control (paper §5.5, App. E).

For a stratified sample of mutants per program, re-initialise the mutant's *own*
compiled architecture (Xavier by default) and train it exactly like the repair run,
for several seeds. The buggy-init side of each pair is the mutant's main repair run.

    python experiments/init_control/launch.py --dry-run
    python -m rasping.results collect init_control --version v2

v1 (TMLR): 20 mutants per program (selection seed 0), seeds 0-4, Xavier; the
selection is reproduced exactly by `select_mutants` (checked by a test).
"""

import json

import click
import numpy as np

from rasping import cluster, raspbugs
from rasping.config import PROGRAMS, RUNS_DIR


def select_mutants(program: str, subset_size: int = 20, seed: int = 0) -> list[str]:
    """Stratified sample by mutation order from the benchmark (manifest order)."""
    ms = raspbugs.mutants(program=program, sort=False)
    rng = np.random.default_rng(seed)
    orders = sorted({m.mutation_order for m in ms})
    per_order = max(1, subset_size // len(orders))
    selected = []
    for o in orders:
        cands = [m.job_id for m in ms if m.mutation_order == o]
        selected.extend(rng.choice(cands, size=min(per_order, len(cands)), replace=False))
    if len(selected) < subset_size:  # top up from the remainder
        rest = [m.job_id for m in ms if m.job_id not in set(selected)]
        if rest:
            selected.extend(rng.choice(rest, size=min(subset_size - len(selected), len(rest)),
                                       replace=False))
    return [str(j) for j in selected[:subset_size]]


@click.command()
@click.option("--subset-size", default=20, show_default=True, help="Mutants per program")
@click.option("--subset-seed", default=0, show_default=True)
@click.option("--seed", "seeds", type=int, multiple=True, default=(0, 1, 2, 3, 4), show_default=True)
@click.option("--init", default="xavier", show_default=True)
@cluster.add_cli_options
def main(subset_size, subset_seed, seeds, init, **cli):
    root = RUNS_DIR / "init_control"
    selection = {p: select_mutants(p, subset_size, subset_seed) for p in PROGRAMS}
    root.mkdir(parents=True, exist_ok=True)
    (root / "selected_jobs.json").write_text(json.dumps(
        {"subset_seed": subset_seed, "subset_size": subset_size, "seeds": list(seeds),
         "init_scheme": init, "selection": selection}, indent=2))
    commands = []
    for program, job_ids in selection.items():
        for job_id in job_ids:
            for seed in seeds:
                out = root / program / job_id / f"{init}_seed_{seed}"
                if not (out / "run.json").exists():
                    commands.append(["python", "-m", "rasping.repair", "--job-id", job_id,
                                     "--out", str(out), "--init", init, "--seed", str(seed)])
    cluster.submit_from_cli(commands, "init_control", **cli)


if __name__ == "__main__":
    main()
