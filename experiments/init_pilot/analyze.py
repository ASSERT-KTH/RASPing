"""Initialization pilot: compare starting points in the compiled architecture.

    python experiments/init_pilot/analyze.py --version v2

Reads results/<version>/init_pilot/runs.csv (collect with
`python -m rasping.results collect init_pilot --version v2`) and writes
    tables/init_pilot_arms.tex     per arm: start agreement, % repaired (all, per program), epochs
    tables/init_pilot_paired.tex   per arm vs compiled and vs xavier: who repairs which mutants
    figures/init_pilot_repaired.pdf  % repaired per program and arm
    tables/init_pilot_patience.tex  per arm: % repaired with patience 50 vs 500 (if those runs exist)

The arm tables use the first round's patience (50) only; the patience-500 runs (`launch.py --patience
500`) appear only in the patience table, paired with the same mutants at patience 50.
"""

import click
import matplotlib.pyplot as plt
import numpy as np

from rasping import analysis
from rasping.config import PROGRAMS, V1_EARLY_STOPPING_PATIENCE
from rasping.results import load_runs
from rasping.tables import latex_table, pct, save_figure, tt, write_table

ARMS = ["compiled", "xavier", "rescaled", "shuffled", "soft:50", "soft:30", "soft:10", "from"]
LABELS = {"compiled": "compiled", "xavier": "Xavier", "rescaled": "rescaled (structure, Xavier scale)",
          "shuffled": "shuffled (values, no structure)", "soft:50": "soft 50", "soft:30": "soft 30",
          "soft:10": "soft 10", "from": "other mutant (same shapes)"}


PATIENCE = V1_EARLY_STOPPING_PATIENCE  # the pilot's first round
HIGH_PATIENCE = 500


def load(version):
    runs = load_runs(version, "init_pilot")
    return runs.assign(arm=runs["init"].map(analysis.init_arm))


def patience_table(runs, version):
    t = analysis.patience_table(runs, PATIENCE, HIGH_PATIENCE, list(PROGRAMS))
    if t.empty:
        return
    print(t.round(1).to_string(index=False))
    rows = [[LABELS[r.arm], r.pairs, f"{pct(r.repaired_lo)} / {pct(r.repaired_hi)}",
             *[f"{pct(getattr(r, f'{p}_lo'))} / {pct(getattr(r, f'{p}_hi'))}" for p in PROGRAMS],
             r.only_hi, r.only_lo, f"{int(r.epochs_lo)} / {int(r.epochs_hi)}"]
            for r in t.itertuples()]
    header = ["Start", "Pairs", "All", *[tt(p) for p in PROGRAMS], f"Only {HIGH_PATIENCE}",
              f"Only {PATIENCE}", "Epochs"]
    write_table(version, "init_pilot_patience.tex", latex_table(header, rows))


def arms_table(t, version):
    rows = [[LABELS[r.arm], r.runs, pct(r.start_agreement), pct(r.behaviour_kept), pct(r.repaired),
             *[pct(getattr(r, f"repaired_{p}")) for p in PROGRAMS], pct(r.test_acc), int(r.median_epochs)]
            for r in t.itertuples()]
    header = ["Start", "Runs", "Agree", "Kept", "All", *[tt(p) for p in PROGRAMS], "Acc.", "Epochs"]
    write_table(version, "init_pilot_arms.tex", latex_table(header, rows))


def paired_table(runs, version):
    rows = []
    for ref in ("compiled", "xavier"):
        for arm in ARMS:
            if arm == ref or arm not in set(runs["arm"]):
                continue
            o = analysis.paired_outcomes(runs, arm, ref)
            rows.append([LABELS[arm], LABELS[ref], o["pairs"], o["only_arm"], o["both"], o["only_ref"], o["neither"]])
    header = ["Start", "vs", "Pairs", "Only start", "Both", "Only vs", "Neither"]
    write_table(version, "init_pilot_paired.tex", latex_table(header, rows))


def figure(t, version):
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.8 / len(t)
    x = np.arange(len(PROGRAMS) + 1)
    for i, r in enumerate(t.itertuples()):
        vals = [getattr(r, f"repaired_{p}") for p in PROGRAMS] + [r.repaired]
        ax.bar(x + (i - len(t) / 2 + 0.5) * width, vals, width, label=LABELS[r.arm])
    ax.set_xticks(x, [*PROGRAMS, "all"])
    ax.set(ylabel="Repaired (%)", ylim=(0, 100))
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, version, "init_pilot_repaired.pdf")


@click.command()
@click.option("--version", default="v2", show_default=True)
def main(version):
    all_runs = load(version)
    runs = all_runs[all_runs["cfg_early_stopping_patience"] == PATIENCE]
    t = analysis.arm_table(runs, ARMS, list(PROGRAMS))
    print(t.round(1).to_string(index=False))
    # Arms whose starting behaviour varies per mutant: repair rate when the start does / does
    # not behave like the compiled mutant.
    kept = runs.assign(kept=runs["start_agreement"] >= 0.99)
    split = kept[kept["arm"].isin(["rescaled", "soft:50", "soft:30", "soft:10"])].groupby(["arm", "kept"])
    print((100 * split["test_accuracy"].apply(lambda a: (a >= 0.99).mean())).round(1).rename("repaired %").to_string())
    arms_table(t, version)
    paired_table(runs, version)
    figure(t, version)
    patience_table(all_runs, version)


if __name__ == "__main__":
    main()
