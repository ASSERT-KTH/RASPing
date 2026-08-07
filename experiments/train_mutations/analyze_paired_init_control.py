"""Paired initialization control: per-mutant scatter of buggy-init vs. random-init
(mutant's own architecture) test accuracy, plus summary table.

Unlike plot_initialization_controls.py (which compares aggregate distributions
against the ground-truth architecture), this is a true paired comparison: each
point is one mutant, evaluated both ways on the identical compiled architecture,
identical data, identical optimizer/early-stopping. Random-init accuracy is the
mean over 5 seeds per mutant.

Usage:
    python analyze_paired_init_control.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).parent
BUGGY_SAVED_DATA = HERE / "saved_data"
PAIRED_SAVED_DATA = HERE / "saved_data_paired_init"
PLOTS_DIR = HERE.parent.parent / "plots"

PROGRAMS = ["hist", "most_freq", "reverse", "sort"]
LOSS_FN = "cross_entropy_loss"


def load_pairs(program: str, selection: list[str]) -> pd.DataFrame:
    rows = []
    for job_id in selection:
        buggy_path = BUGGY_SAVED_DATA / program / LOSS_FN / f"job_{job_id}" / "test_results.json"
        if not buggy_path.exists():
            continue
        buggy_acc = json.load(open(buggy_path))["test_accuracy"]

        base = PAIRED_SAVED_DATA / program / f"job_{job_id}"
        accs, epochs = [], []
        for seed_dir in sorted(base.glob("seed_*")):
            tr = seed_dir / LOSS_FN / "test_results.json"
            tl = seed_dir / LOSS_FN / "train_losses.npy"
            if tr.exists():
                accs.append(json.load(open(tr))["test_accuracy"])
            if tl.exists():
                epochs.append(len(np.load(tl, allow_pickle=True)))
        if not accs:
            continue

        rows.append({
            "program": program,
            "job_id": job_id,
            "buggy_init_acc": buggy_acc,
            "random_init_acc_mean": float(np.mean(accs)),
            "random_init_acc_min": float(np.min(accs)),
            "random_init_acc_max": float(np.max(accs)),
            "random_init_n_seeds": len(accs),
            "random_init_epochs_mean": float(np.mean(epochs)) if epochs else float("nan"),
        })
    return pd.DataFrame(rows)


def classify(df: pd.DataFrame, margin: float = 0.01) -> pd.DataFrame:
    diff = df["random_init_acc_mean"] - df["buggy_init_acc"]
    df = df.copy()
    df["outcome"] = np.select(
        [diff > margin, diff < -margin],
        ["random_wins", "buggy_wins"],
        default="tie",
    )
    return df


def make_scatter(df: pd.DataFrame, out_path: Path):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

    palette = {"random_wins": "#C44E52", "tie": "#8C8C8C", "buggy_wins": "#4C72B0"}

    for ax, program in zip(axes, PROGRAMS):
        sub = df[df["program"] == program]
        for outcome, color in palette.items():
            s = sub[sub["outcome"] == outcome]
            ax.scatter(s["buggy_init_acc"] * 100, s["random_init_acc_mean"] * 100,
                       color=color, s=40, alpha=0.8, label=outcome.replace("_", " "))
        ax.plot([0, 100], [0, 100], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Buggy-init test acc. (%)")
        ax.set_title(program)

    axes[0].set_ylabel("Random-init test acc. (%, mean of 5 seeds)")
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Paired initialization control: same mutant architecture, buggy vs. random weights")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {out_path} (+ .png)")


def make_latex_table(summary: pd.DataFrame, out_path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired initialization control: each mutant's own compiled architecture, "
        r"repaired from the buggy checkpoint vs. randomly initialized (xavier, mean of 5 "
        r"seeds), identical data/optimizer/early-stopping otherwise.}",
        r"\label{tab:paired-init-control}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Program & $n$ & Buggy mean & Random mean & Random wins & Ties & Buggy wins \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['program']} & {row['n']} & {row['buggy_mean']*100:.1f}\\% & "
            f"{row['random_mean']*100:.1f}\\% & {row['random_wins']} & {row['ties']} & "
            f"{row['buggy_wins']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


def main():
    selection = json.load(open(PAIRED_SAVED_DATA / "selected_jobs.json"))["selection"]

    all_rows = []
    for program in PROGRAMS:
        df = load_pairs(program, selection[program])
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    df = classify(df)

    df.to_csv(HERE / "paired_init_control_pairs.csv", index=False)
    print(f"Saved {HERE / 'paired_init_control_pairs.csv'} ({len(df)} rows)")

    summary_rows = []
    for program in PROGRAMS:
        sub = df[df["program"] == program]
        summary_rows.append({
            "program": program,
            "n": len(sub),
            "buggy_mean": sub["buggy_init_acc"].mean(),
            "buggy_median": sub["buggy_init_acc"].median(),
            "random_mean": sub["random_init_acc_mean"].mean(),
            "random_median": sub["random_init_acc_mean"].median(),
            "random_wins": (sub["outcome"] == "random_wins").sum(),
            "ties": (sub["outcome"] == "tie").sum(),
            "buggy_wins": (sub["outcome"] == "buggy_wins").sum(),
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(HERE / "paired_init_control_summary.csv", index=False)

    make_scatter(df, PLOTS_DIR / "paired_init_control.pdf")
    make_latex_table(summary, HERE / "paired_init_control_table.tex")


if __name__ == "__main__":
    main()
