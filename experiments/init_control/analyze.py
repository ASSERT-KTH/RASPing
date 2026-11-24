"""Initialization control: Table 2, Figure 9, Figure 10, Table 9 (paper §5.5, App. E).

    python experiments/init_control/analyze.py --version v1_tmlr

Pairs each selected mutant's main repair run (compiled buggy weights) with its
random-init runs (mean over seeds). Table 9 needs weight statistics, computed from raw
run directories by `experiments/init_control/weight_stats.py`.
"""

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rasping import analysis
from rasping.config import PROGRAMS, TrainConfig
from rasping.results import load_runs, results_dir
from rasping.tables import latex_table, pct, save_figure, tt, write_table

PALETTE = {"random_wins": "#C44E52", "tie": "#8C8C8C", "buggy_wins": "#4C72B0"}


def half_up(x):
    return int(np.floor(x + 0.5))


def table2(summary, version):
    rows = []
    for r in summary.itertuples():
        name = r"\textbf{All}" if r.program == "All" else tt(r.program)
        rows.append([name, r.n, pct(r.buggy_acc), pct(r.random_acc), pct(r.buggy_repaired),
                     pct(r.random_repaired), half_up(r.buggy_epochs), half_up(r.random_epochs)])
    header = ["Program", "$n$", "buggy", "random", "buggy", "random", "buggy", "random"]
    tex = latex_table(header, rows, align="lrrrrrrr", midrule_before_last=True)
    group = (r" & & \multicolumn{2}{c}{Mean test accuracy} & \multicolumn{2}{c}{Repaired ($\geq$99\%)}"
             r" & \multicolumn{2}{c}{Epochs to early stop} \\" "\n"
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n")
    tex = tex.replace(r"\toprule" + "\n", r"\toprule" + "\n" + group, 1)
    write_table(version, "table2_init_control.tex", tex)


def figure9(pm, version):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    for ax, program in zip(axes, PROGRAMS):
        sub = pm[pm["program"] == program]
        for outcome, color in PALETTE.items():
            s = sub[sub["outcome"] == outcome]
            ax.scatter(100 * s["buggy_acc"], 100 * s["random_acc"], color=color, s=55, alpha=0.8,
                       label=outcome.replace("_", " "))
        ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5)
        ax.set(xlim=(-3, 103), ylim=(-3, 103), xlabel="Buggy-init test acc. (%)", title=program)
    axes[0].set_ylabel("Random-init test acc. (%)")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, version, "fig9_init_control_pairs.pdf")


def _held_matrix(val: pd.DataFrame, n: int) -> np.ndarray:
    """Runs x `n` evaluations, each run held at its final value after it stops."""
    series = [g.sort_values("val_eval")["val_acc"].to_numpy() for _, g in val.groupby(RUN)]
    return np.array([np.concatenate([s, np.full(n - len(s), s[-1])]) for s in series])


RUN = ["job_id", "seed"]


def figure10(buggy_val, random_val, version):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, program in zip(axes, PROGRAMS):
        # each panel spans the program's longest run (of either initialization)
        n = int(max(v[v["program"] == program]["val_eval"].max() for v in (buggy_val, random_val))) + 1
        for label, val, color in (("buggy init", buggy_val, "tab:blue"),
                                  ("random init", random_val, "tab:orange")):
            m = 100 * _held_matrix(val[val["program"] == program], n)
            x = np.arange(m.shape[1]) * TrainConfig.val_every
            ax.plot(x, np.median(m, axis=0), color=color, label=label)
            ax.fill_between(x, *np.percentile(m, [25, 75], axis=0), color=color, alpha=0.2)
        ax.set(title=program, xlabel="Epoch", ylim=(0, 102))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, version, "fig10_init_control_curves.pdf")


def table9(version):
    path = results_dir(version, "init_control") / "weight_stats.csv"
    if not path.exists():
        print(f"skipping Table 9: {path} not found (run weight_stats.py)")
        return
    t = analysis.weight_stats_ranges(pd.read_csv(path)).set_index(["program", "state"])
    states = ["compiled", "repaired_from_compiled", "trained_from_xavier"]
    rows = []
    for p in PROGRAMS:
        r = [t.loc[(p, s)] for s in states]
        rows.append([tt(p),
                     *[f"{x.zero_min:.1f}--{x.zero_max:.1f}\\%" for x in r],
                     *[f"{x.std_min:.2f}--{x.std_max:.2f}" for x in r],
                     *[f"{x.max_abs:.2f}" for x in r]])
    sub = ["compiled", "repaired", "from Xavier"] * 3
    write_table(version, "table9_weight_stats.tex", latex_table(["Program", *sub], rows))


@click.command()
@click.option("--version", default="v1_tmlr", show_default=True)
def main(version):
    main_runs = load_runs(version, "main_repair")
    rnd = load_runs(version, "init_control")
    pm, summary = analysis.paired_init(main_runs, rnd)
    print(summary.round(1).to_string(index=False))
    table2(summary, version)
    figure9(pm, version)

    vb = pd.read_csv(results_dir(version, "main_repair") / "val_curves.csv.gz")
    vr = pd.read_csv(results_dir(version, "init_control") / "val_curves.csv.gz")
    prog = main_runs.set_index("job_id")["program"]
    vb = vb[vb["job_id"].isin(pm["job_id"])].assign(seed=-1)
    vb["program"] = vb["job_id"].map(prog)
    vr["program"] = vr["job_id"].map(prog)
    figure10(vb, vr, version)
    table9(version)


if __name__ == "__main__":
    main()
