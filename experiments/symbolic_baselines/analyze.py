"""Repair rate over search progress, GBPR vs GP vs BFS: Figure 4 (steps) and Figure 5 (time).

    python experiments/symbolic_baselines/analyze.py --version v1_tmlr

One panel per mutation order, over the main-evaluation mutants that all three methods
have results for. Curve semantics are those of the v1 paper (see FINDINGS):
  * GP/BFS: best *test* accuracy over all candidates evaluated so far (F17);
    x = programs evaluated; time = constant duration_s / num_programs per job.
  * GBPR: best-so-far *validation* accuracy (F4), one point per validation evaluation,
    i.e. every 10 epochs (F16); time = per-program seconds per validation step from
    gbpr_timing.json in v1 (F18), measured per run in later versions.
"""

import json

import click
import matplotlib.pyplot as plt
import pandas as pd

from rasping import analysis
from rasping.config import MAIN_PROGRAMS, METHOD_LABEL, TrainConfig
from rasping.results import load_runs, results_dir
from rasping.tables import save_figure

COLORS = {"GP": "#1f77b4", METHOD_LABEL: "#2ca02c", "BFS": "#ff7f0e"}


def load_series(version: str, budget: int):
    main = load_runs(version, "main_repair")
    main = main[main["program"].isin(MAIN_PROGRAMS)]
    val = pd.read_csv(results_dir(version, "main_repair") / "val_curves.csv.gz")
    series = {METHOD_LABEL: analysis.cummax_series(val[val["job_id"].isin(main["job_id"])])}
    sec_per_step = {}

    timing = results_dir(version, "symbolic_baselines") / "gbpr_timing.json"
    if timing.exists():  # v1: per-program benchmark
        t = json.loads(timing.read_text())
        prog = main.set_index("job_id")["program"]
        sec_per_step[METHOD_LABEL] = {j: t[prog[j]]["seconds_per_epoch"] * TrainConfig.val_every
                                      for j in series[METHOD_LABEL]}
    else:  # measured: mean epoch time of the run x validation interval
        sec_per_step[METHOD_LABEL] = dict(zip(
            main["job_id"], main["wall_time_s"] / main["n_epochs"] * main["cfg_val_every"]))

    for method in ("gp", "bfs"):
        d = results_dir(version, "symbolic_baselines")
        runs = load_runs(version, "symbolic_baselines", f"{method}_budget{budget}_runs.csv")
        hist = pd.read_csv(d / f"{method}_budget{budget}_history.csv.gz")
        name = method.upper()
        series[name] = analysis.best_so_far_series(hist)
        sec_per_step[name] = dict(zip(runs["job_id"], runs["duration_s"] / runs["num_programs"]))

    common = set.intersection(*(set(s) for s in series.values()))
    series = {m: {j: s[j] for j in common} for m, s in series.items()}
    order = main.set_index("job_id")["mutation_order"]
    return series, sec_per_step, order


def plot(series, sec_per_step, order, time_axis: bool, version: str, name: str):
    orders = sorted(order[list(next(iter(series.values())))].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, o in zip(axes.flat, orders):
        x_max, curves = 0, {}
        for m, s in series.items():
            sub = {j: v for j, v in s.items() if order[j] == o}
            x, y = (analysis.pass_rate_by_time(sub, sec_per_step[m]) if time_axis
                    else analysis.pass_rate_by_step(sub))
            curves[m] = (x, y)
            x_max = max(x_max, x[-1])
        for m in ("GP", METHOD_LABEL, "BFS"):
            x, y = curves[m]
            ax.plot(x, y, label=m, color=COLORS[m])
            if x[-1] < x_max:  # extend shorter curves with a dotted tail
                ax.plot([x[-1], x_max], [y[-1], y[-1]], color=COLORS[m], ls="--", dashes=(1, 1))
        n = sum(1 for j in series["GP"] if order[j] == o)
        ax.set_title(f"Mutation Order: {o} ({n} buggy mutants)")
        ax.set_xlabel("Time (s)" if time_axis else "Steps")
        ax.set_ylabel("% fixed programs")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
    for ax in list(axes.flat)[len(orders):]:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, version, name)


@click.command()
@click.option("--version", default="v1_tmlr", show_default=True)
@click.option("--budget", default=1000, show_default=True)
def main(version, budget):
    series, sps, order = load_series(version, budget)
    print(f"{len(series['GP'])} mutants with results for all methods")
    plot(series, sps, order, False, version, "fig4_pass_rate_by_steps.pdf")
    plot(series, sps, order, True, version, "fig5_pass_rate_by_time.pdf")


if __name__ == "__main__":
    main()
