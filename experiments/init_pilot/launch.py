"""Initialization pilot: which starting point in the compiled architecture trains best?

For a stratified sample of mutants per program, repair each mutant from several starting
points, all in the mutant's own compiled architecture (see `rasping.repair.INITS`):

    compiled     the mutant's Tracr weights (the method)
    xavier       random re-initialisation
    rescaled     compiled structure at Xavier scale        } with compiled and xavier, a 2x2:
    shuffled     compiled values permuted per tensor       } structure x scale
    soft:50/30/10  softer compilation that approximates the mutant (keeps some of its behaviour)
    from         another mutant's compiled weights with identical shapes (if one exists)

Every run records `start_agreement`: the fraction of test inputs on which the starting
weights produce exactly the compiled mutant's output. Probes on 24 mutants (Nov 24, 2026):
- **rescaled** often keeps the mutant's behaviour completely. Per mutant it is bimodal
  (agreement ~0 or 1); the mean is 0.36–0.83 depending on the program. So it is the buggy
  program at Xavier scale more often than not, and results should be split by agreement.
- **soft 30 / soft 10** keep 0.68–0.91 / 0.39–0.85 on average.
- **shuffled and xavier** keep ~0.
- **uniform rescaling** of all weights (x0.9 … x0.03) was considered and dropped. Even x0.9
  loses up to two thirds of the behaviour, while x0.3 keeps none.

    python experiments/init_pilot/launch.py --dry-run

Early-stopping check (Nov 25, 2026): in the first round every most_freq run stopped early
just below the threshold (median test accuracy 0.88-0.96 for xavier, rescaled and soft:10).
Rerun four arms with patience 500 instead of 50; these runs go to
`<arm>_seed_<s>_patience500` next to the patience-50 runs, and the analysis pairs them:

    python experiments/init_pilot/launch.py --patience 500 \
        --arm compiled --arm xavier --arm rescaled --arm soft:10
    python -m rasping.results collect init_pilot --version v2
    python experiments/init_pilot/analyze.py --version v2
"""

import json

import click
import numpy as np

from rasping import cluster, raspbugs
from rasping.config import PROGRAMS, RUNS_DIR, V1_EARLY_STOPPING_PATIENCE

ARMS = ("compiled", "xavier", "rescaled", "shuffled", "soft:50", "soft:30", "soft:10", "from")


def pick_twin(job_id: str, seed: int) -> str | None:
    """A shape twin of `job_id` chosen with a per-mutant seeded draw (None if there is none)."""
    twins = raspbugs.shape_twins(job_id)
    if not twins:
        return None
    rng = np.random.default_rng([seed, int(job_id[:8], 16)])
    return str(rng.choice(twins))


@click.command()
@click.option("--n-per-program", default=25, show_default=True)
@click.option("--sample-seed", default=0, show_default=True)
@click.option("--seed", "seeds", type=int, multiple=True, default=(0,), show_default=True)
@click.option("--arm", "arms", multiple=True, default=ARMS, show_default=True,
              type=click.Choice(ARMS))
@click.option("--program", "programs", multiple=True, default=PROGRAMS, show_default=True,
              type=click.Choice(PROGRAMS))
@click.option("--patience", type=int, default=V1_EARLY_STOPPING_PATIENCE, show_default=True,
              help="Early-stopping patience; other values than the first round's 50 get a "
                   "_patience<P> run directory")
@cluster.add_cli_options
def main(n_per_program, sample_seed, seeds, arms, programs, patience, **cli):
    root = RUNS_DIR / "init_pilot"
    default = patience == V1_EARLY_STOPPING_PATIENCE
    suffix = "" if default else f"_patience{patience}"
    selection = {p: raspbugs.stratified_sample(p, n_per_program, sample_seed) for p in programs}
    twins = {j: pick_twin(j, sample_seed) for js in selection.values() for j in js}
    if not cli["dry_run"]:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"selected_jobs{suffix}.json").write_text(json.dumps(
            {"sample_seed": sample_seed, "n_per_program": n_per_program, "seeds": list(seeds),
             "arms": list(arms), "patience": patience, "selection": selection, "twins": twins},
            indent=2))
    commands = []
    for program, job_ids in selection.items():
        for job_id in job_ids:
            for arm in arms:
                if arm == "from":
                    if twins[job_id] is None:
                        continue
                    init = f"from:{twins[job_id]}"
                else:
                    init = arm
                for seed in seeds:
                    out = root / program / job_id / f"{arm.replace(':', '_')}_seed_{seed}{suffix}"
                    if not (out / "run.json").exists():
                        commands.append(["python", "-m", "rasping.repair", "--job-id", job_id,
                                         "--out", str(out), "--init", init, "--seed", str(seed),
                                         "--patience", str(patience)])
    n_twinless = sum(t is None for t in twins.values())
    print(f"{sum(map(len, selection.values()))} mutants x {len(arms)} arms x {len(seeds)} seeds; "
          f"{n_twinless} mutants have no shape twin (no 'from' run); {len(commands)} runs to submit")
    cluster.submit_from_cli(commands, f"init_pilot{suffix}", **cli)


if __name__ == "__main__":
    main()
