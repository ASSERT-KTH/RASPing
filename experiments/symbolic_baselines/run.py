"""Run one symbolic repair baseline (GP or BFS) on one RaspBugs mutant.

Writes the result record to `<runs-dir>/symbolic_baselines/<method>_budget<B>/<job_id>.json`:
the v1 keys (status, num_programs, num_duplicates, num_compile_failures, train_acc,
test_acc, best_source, program_history, job_id, program_name, duration_s, and seed for
GP) plus method, budget, program and the search hyperparameters.

    python experiments/symbolic_baselines/run.py --method gp --job-id <id> --budget 1000
    python experiments/symbolic_baselines/run.py --method bfs --job-id <id> --budget 1000

Defaults are the v1 (TMLR) settings: GP population 16, tournament 3, seed 42; BFS depth
= the mutant's mutation order (docs/FINDINGS.md F3); 30 s evaluation timeout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from rasping import raspbugs
from rasping.baselines import METHODS, bfs, gp, run_dir
from rasping.config import DATA_DIR, RUNS_DIR


@click.command()
@click.option("--method", type=click.Choice(METHODS), required=True)
@click.option("--job-id", required=True, help="RaspBugs mutant id")
@click.option("--budget", type=int, default=1000, show_default=True,
              help="Number of generated programs (duplicates and the mutant itself included)")
@click.option("--population-size", type=int, default=gp.DEFAULTS["population_size"],
              show_default=True, help="GP: population size mu = offspring per generation lambda")
@click.option("--tournament-k", type=int, default=gp.DEFAULTS["tournament_k"], show_default=True,
              help="GP: tournament size")
@click.option("--seed", type=int, default=gp.DEFAULTS["seed"], show_default=True, help="GP: RNG seed")
@click.option("--max-depth", type=int, default=None,
              help="BFS: search depth [default: the mutant's mutation order]")
@click.option("--eval-timeout", type=float, default=30.0, show_default=True,
              help="Seconds allowed for evaluating one candidate on one dataset")
@click.option("--log-every", type=int, default=None, help="Progress log interval [GP 25, BFS 50]")
@click.option("--data-dir", type=click.Path(path_type=Path), default=DATA_DIR, show_default=True)
@click.option("--runs-dir", type=click.Path(path_type=Path), default=RUNS_DIR, show_default=True)
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(method, job_id, budget, population_size, tournament_k, seed, max_depth, eval_timeout,
         log_every, data_dir, runs_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level),
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    mutant = raspbugs.get(job_id)
    if method == "gp":
        result = gp.run_gp_for_mutant(
            mutant, data_dir, budget=budget, population_size=population_size,
            tournament_k=tournament_k, seed=seed, eval_timeout=eval_timeout,
            log_every=25 if log_every is None else log_every)
    else:
        result = bfs.run_bfs_for_mutant(
            mutant, data_dir, max_depth=max_depth, budget=budget, eval_timeout=eval_timeout,
            log_every=50 if log_every is None else log_every)

    out = run_dir(method, budget, runs_dir) / f"{job_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(out)
    print(json.dumps({k: result.get(k) for k in
                      ("job_id", "program", "method", "status", "num_programs", "train_acc",
                       "test_acc", "duration_s")}))


if __name__ == "__main__":
    main()
