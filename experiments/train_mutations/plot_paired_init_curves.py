"""Learning curves for the paired initialization control.

For each of the 80 selected mutants we have one buggy-init run and 5 random-init
runs on the same compiled architecture. This plots validation accuracy against
epoch, aggregated across mutants: each run is held at its final value after its
own early stop, so the aggregate curve reflects both speed and final quality.

Usage:
    python plot_paired_init_curves.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).parent
BUGGY_SAVED_DATA = HERE / "saved_data"
PAIRED_SAVED_DATA = HERE / "saved_data_paired_init"
PLOTS_DIR = HERE.parent.parent / "plots"

PROGRAMS = ["hist", "most_freq", "reverse", "sort"]
LOSS_FN = "cross_entropy_loss"
VAL_STEP = 10  # train_mutations.py records validation accuracy every 10 epochs

BUGGY_COLOR = "#0072B2"   # Okabe-Ito blue
RANDOM_COLOR = "#D55E00"  # Okabe-Ito vermillion


def pad(curve: np.ndarray, n: int) -> np.ndarray:
    """Hold the last value after early stopping, so all curves share a length.

    Never truncates: n is chosen per panel as the longest run in that panel, so a
    curve is only ever extended. Truncating would make a still-improving run look
    like it had plateaued.
    """  # noqa: D401
    curve = np.asarray(curve, dtype=float).ravel()
    if len(curve) < n:
        curve = np.concatenate([curve, np.full(n - len(curve), curve[-1])])
    return curve


def main():
    selection = json.load(open(PAIRED_SAVED_DATA / "selected_jobs.json"))["selection"]
    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)  # x differs per panel

    for ax, program in zip(axes, PROGRAMS):
        buggy, random = [], []
        for job_id in selection[program]:
            bp = BUGGY_SAVED_DATA / program / LOSS_FN / f"job_{job_id}" / "val_accs.npy"
            if not bp.exists():
                continue
            base = PAIRED_SAVED_DATA / program / f"job_{job_id}"
            seeds = [s / LOSS_FN / "val_accs.npy" for s in sorted(base.glob("seed_*"))]
            seeds = [s for s in seeds if s.exists()]
            if not seeds:
                continue
            buggy.append(np.load(bp, allow_pickle=True).ravel())
            random.extend(np.load(s, allow_pickle=True).ravel() for s in seeds)

        # Every run is shown to its own early stop; the panel spans the longest one.
        n_epochs = max(len(c) for c in buggy + random)
        buggy = [pad(c, n_epochs) for c in buggy]
        random = [pad(c, n_epochs) for c in random]

        x = np.arange(1, n_epochs + 1) * VAL_STEP
        for curves, color, label in ((buggy, BUGGY_COLOR, "buggy init"),
                                     (random, RANDOM_COLOR, "random init")):
            arr = np.vstack(curves) * 100
            med = np.median(arr, axis=0)
            lo, hi = np.percentile(arr, [25, 75], axis=0)
            ax.plot(x, med, color=color, linewidth=2, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)

        ax.set_xlabel("Epoch")
        ax.set_title(program)
        ax.set_xlim(VAL_STEP, n_epochs * VAL_STEP)
        ax.set_ylim(-3, 103)

    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "paired_init_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {out} (+ .png)")


if __name__ == "__main__":
    main()
