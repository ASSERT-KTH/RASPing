"""Repair rate over search progress, GBPR vs GP vs BFS: Figure 4 (steps) and Figure 5 (time).

v2 results (selected-program test accuracy + measured timestamps) are used when present;
otherwise the v1 semantics below reproduce the submitted figures.

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
import numpy as np
import pandas as pd

from rasping import analysis
from rasping.config import PROGRAMS, METHOD_LABEL, TrainConfig
from rasping.results import load_runs, results_dir
from rasping.tables import save_figure

COLORS = {"GP": "#1f77b4", METHOD_LABEL: "#2ca02c", "BFS": "#ff7f0e"}


def _const_rate(series, sec_per_step):
    """(times, values) for a constant time per step, as the v1 figures assumed: value k is
    shown from k*sec_per_step and the last one is held until len*sec_per_step."""
    t = np.arange(len(series) + 1) * sec_per_step
    return t, list(series) + [series[-1]]


def load_series(version: str, budget: int):
    """Per-method, per-job curves on a step axis and on a wall-clock axis.

    Returns (step_series, time_series, order): step_series[m][job] is a list over steps;
    time_series[m][job] is (times, values).
    """
    main = load_runs(version, "main_repair")
    main = main[main["program"].isin(PROGRAMS)]
    d = results_dir(version, "main_repair")
    val = pd.read_csv(d / "val_curves.csv.gz")
    val = val[val["job_id"].isin(main["job_id"])]
    measured = "selected_test_acc" in val.columns and val["selected_test_acc"].notna().any()

    steps, times = {}, {}
    if measured:  # v2: test accuracy of the selected (best-val-loss) params, measured time
        steps[METHOD_LABEL] = {j: g.sort_values("val_eval")["selected_test_acc"].tolist()
                               for j, g in val.groupby("job_id")}
        curves = pd.read_csv(d / "curves.csv.gz")
        every = dict(zip(main["job_id"], main["cfg_val_every"]))
        times[METHOD_LABEL] = {}
        for j, g in curves[curves["job_id"].isin(steps[METHOD_LABEL])].groupby("job_id"):
            t = g.sort_values("epoch")["time_s"].to_numpy()
            ev = np.arange(len(steps[METHOD_LABEL][j])) * every[j]
            times[METHOD_LABEL][j] = (t[np.minimum(ev, len(t) - 1)], steps[METHOD_LABEL][j])
    else:  # v1: best-so-far validation accuracy (F4), benchmark-estimated time (F18)
        steps[METHOD_LABEL] = analysis.cummax_series(val)
        t = json.loads((results_dir(version, "symbolic_baselines") / "gbpr_timing.json").read_text())
        prog = main.set_index("job_id")["program"]
        times[METHOD_LABEL] = {
            j: _const_rate(s, t[prog[j]]["seconds_per_epoch"] * TrainConfig.val_every)
            for j, s in steps[METHOD_LABEL].items()}

    for method in ("gp", "bfs"):
        bd = results_dir(version, "symbolic_baselines")
        runs = load_runs(version, "symbolic_baselines", f"{method}_budget{budget}_runs.csv")
        hist = pd.read_csv(bd / f"{method}_budget{budget}_history.csv.gz")
        name = method.upper()
        has_time = "time_s" in hist.columns and hist["time_s"].notna().any()
        steps[name] = analysis.best_so_far_series(hist, running_max=not has_time)
        if has_time:
            times[name] = {j: (g["time_s"].to_numpy(), g["best_test_acc"].to_numpy())
                           for j, g in hist.groupby("job_id")}
        else:  # v1: constant seconds per evaluated program
            sps = dict(zip(runs["job_id"], runs["duration_s"] / runs["num_programs"]))
            times[name] = {j: _const_rate(s, sps[j]) for j, s in steps[name].items()}

    common = set.intersection(*(set(s) for s in steps.values()))
    steps = {m: {j: s[j] for j in common} for m, s in steps.items()}
    times = {m: {j: s[j] for j in common} for m, s in times.items()}
    order = main.set_index("job_id")["mutation_order"]
    return steps, times, order


def plot(series, order, time_axis: bool, version: str, name: str):
    orders = sorted(order[list(next(iter(series.values())))].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, o in zip(axes.flat, orders):
        x_max, curves = 0, {}
        for m, s in series.items():
            sub = {j: v for j, v in s.items() if order[j] == o}
            x, y = (analysis.pass_rate_on_time_grid(sub) if time_axis
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
        ax.set_xlabel("Time (s)" if time_axis
                      else f"Evaluations ({METHOD_LABEL}: every {TrainConfig.val_every} epochs; GP/BFS: programs)")
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
    steps, times, order = load_series(version, budget)
    print(f"{len(steps['GP'])} mutants with results for all methods")
    plot(steps, order, False, version, "fig4_pass_rate_by_steps.pdf")
    plot(times, order, True, version, "fig5_pass_rate_by_time.pdf")


if __name__ == "__main__":
    main()
