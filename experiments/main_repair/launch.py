"""Launch gradient-based repair on every mutant of a benchmark (paper §5.2; Table 1, Figs 3/7).

One SLURM job per mutant, start and seed, each running `python -m rasping.repair` in the
container and writing `runs/main_repair/<program>/<job_id>/<init>_seed_<s>/` (`soft:10` is
written `soft_10`). Already finished runs are skipped, so re-running the launcher
resubmits only missing runs. With `--benchmark tracrbench`, runs go to
`runs/tracrbench_repair/` (collect with `python -m rasping.results collect tracrbench_repair`).

    python experiments/main_repair/launch.py --dry-run          # show the commands
    python experiments/main_repair/launch.py                    # submit (v2: 3 starts x 3 seeds)
    python -m rasping.results collect main_repair --version v2  # then collect

v2 (Nov 26, 2026): starts compiled, xavier and rescaled (`config.V2_INITS`, chosen from
the init pilot, `experiments/init_pilot`), seeds 0-2, `TrainConfig` (patience 500, F2). GPU runs
are not reproducible one by one (F39), so results are reported per mutant over seeds.

v1 (TMLR) runs: the 1,466 v1 mutants (Dyck programs included), compiled start only, one
run each, 10k epochs, batch 256, lr 1e-4, patience 50, trajectories stored every 50 steps.
"""

import click

from rasping import cluster
from rasping.benchmarks import BENCHMARKS, REPAIR_EXPERIMENT
from rasping.config import RUNS_DIR, V2_INITS, V2_SEEDS
from rasping.repair import check_init


def run_dir(program: str, job_id: str, init: str, seed: int, benchmark: str = "raspbugs"):
    return (RUNS_DIR / REPAIR_EXPERIMENT[benchmark] / program / job_id
            / f"{init.replace(':', '_')}_seed_{seed}")


def _check_inits(ctx, param, values):
    try:
        return tuple(check_init(v) for v in values)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


@click.command()
@click.option("--benchmark", type=click.Choice(sorted(BENCHMARKS)), default="raspbugs", show_default=True,
              help="tracrbench runs go to runs/tracrbench_repair/")
@click.option("--init", "inits", multiple=True, default=V2_INITS, show_default=True, callback=_check_inits,
              help="Starting point (repeatable)")
@click.option("--seed", "seeds", type=int, multiple=True, default=V2_SEEDS, show_default=True)
@click.option("--program", "programs", multiple=True, help="Only these programs (default: all)")
@click.option("--n-per-program", type=int, default=None,
              help="Only a sample of this many mutants per program, stratified by order (pilots)")
@click.option("--sample-seed", type=int, default=0, show_default=True)
@click.option("--timeout-hours", type=float, default=24, show_default=True,
              help="SLURM time limit per run; run.json is written only at the end")
@click.option("--extra", default="", help="Extra arguments passed to rasping.repair")
@cluster.add_cli_options
def main(benchmark, inits, seeds, programs, n_per_program, sample_seed, timeout_hours, extra, **cli):
    bench = BENCHMARKS[benchmark]
    unknown = set(programs) - set(bench.programs())
    if unknown:
        raise click.BadParameter(f"not {benchmark} programs: {sorted(unknown)}", param_hint="--program")
    names = list(programs) or bench.programs()
    if n_per_program is None:
        mutants = [m for m in bench.mutants() if m.program in names]
    else:
        mutants = [bench.get(j) for p in names for j in bench.stratified_sample(p, n_per_program, sample_seed)]
    commands = []
    for m in mutants:
        for init in inits:
            for seed in seeds:
                out = run_dir(m.program, m.job_id, init, seed, benchmark)
                if not (out / "run.json").exists():
                    commands.append(["python", "-m", "rasping.repair", "--job-id", m.job_id,
                                     "--out", str(out), "--init", init, "--seed", str(seed),
                                     *extra.split()])
    total = len(mutants) * len(inits) * len(seeds)
    print(f"{len(mutants)} mutants x {len(inits)} starts x {len(seeds)} seeds = {total} runs; "
          f"{total - len(commands)} done, {len(commands)} to submit")
    cluster.submit_from_cli(commands, REPAIR_EXPERIMENT[benchmark], timeout_min=int(timeout_hours * 60), **cli)


if __name__ == "__main__":
    main()
