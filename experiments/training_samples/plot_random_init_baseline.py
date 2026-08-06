"""Plot + table for the random-weight-initialization training baseline.

Loads saved_data/{program}/{n_samples}/{metrics,validations}.npy for the
non-Dyck programs (hist, most-freq, reverse, sort) and produces a
publication-ready figure and LaTeX table of final validation (full-sequence
exact-match) accuracy as a function of training-set size.

Dyck programs are excluded here: their validation accuracy uses the same
naive full-sequence exact-match metric shown elsewhere in the paper to be
inflated by severe class imbalance, so they need the balanced-accuracy
treatment before being reported alongside these numbers.
"""
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).parent
SAVED_DATA = HERE / "saved_data"
PLOTS_DIR = HERE.parent.parent / "plots"

PROGRAMS = ["hist", "most-freq", "reverse", "sort"]
PROGRAM_LABELS = {
    "hist": "hist",
    "most-freq": "most-freq",
    "reverse": "reverse",
    "sort": "sort",
}
SAMPLE_SIZES = [5000, 25000, 50000, 100000]


def load_results():
    rows = []
    for program in PROGRAMS:
        for n_samples in SAMPLE_SIZES:
            path = SAVED_DATA / program / str(n_samples)
            metrics_path = path / "metrics.npy"
            validations_path = path / "validations.npy"
            if not metrics_path.exists() or not validations_path.exists():
                continue
            metrics = np.load(metrics_path, allow_pickle=True)
            validations = np.load(validations_path, allow_pickle=True)
            rows.append({
                "program": program,
                "n_samples": n_samples,
                "final_loss": float(metrics[-1]["loss"]),
                "final_val_acc": float(validations[-1]) if len(validations) else float("nan"),
                "best_val_acc": float(np.max(validations)) if len(validations) else float("nan"),
            })
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_path: Path):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    palette = sns.color_palette("colorblind", n_colors=len(PROGRAMS))
    for color, program in zip(palette, PROGRAMS):
        sub = df[df["program"] == program].sort_values("n_samples")
        if sub.empty:
            continue
        ax.plot(
            sub["n_samples"], sub["final_val_acc"] * 100,
            marker="o", label=PROGRAM_LABELS[program], color=color, linewidth=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Learning from random initialization")
    ymax = max(df["final_val_acc"].max() * 100 * 1.25, 1.0)
    ax.set_ylim(-ymax * 0.03, ymax)
    ax.legend(title="Program", loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path} (+ .png)")


def make_latex_table(df: pd.DataFrame, out_path: Path):
    pivot = df.pivot(index="program", columns="n_samples", values="final_val_acc")
    pivot = pivot.reindex(PROGRAMS)
    pivot = pivot.reindex(columns=SAMPLE_SIZES)

    col_headers = " & ".join(f"{n // 1000}k" for n in SAMPLE_SIZES)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Validation accuracy (full-sequence exact match) when training from a random "
        r"weight initialization, as a function of training-set size.}",
        r"\label{tab:random-init-baseline}",
        r"\begin{tabular}{l" + "r" * len(SAMPLE_SIZES) + "}",
        r"\toprule",
        r"Program & " + col_headers + r" \\",
        r"\midrule",
    ]
    for program in PROGRAMS:
        cells = []
        for n in SAMPLE_SIZES:
            val = pivot.loc[program, n] if program in pivot.index else float("nan")
            cells.append("--" if pd.isna(val) else f"{val * 100:.1f}\\%")
        lines.append(f"{PROGRAM_LABELS[program]} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {out_path}")


def main():
    df = load_results()
    print(df.to_string(index=False))

    make_plot(df, PLOTS_DIR / "random_init_baseline.pdf")
    make_latex_table(df, HERE / "random_init_baseline_table.tex")

    csv_path = HERE / "random_init_baseline.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw numbers to {csv_path}")


if __name__ == "__main__":
    main()
