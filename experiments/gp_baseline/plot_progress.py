import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def load_gp_accuracies(results_dir: Path) -> tuple[List[float], List[int]]:
    files = sorted(results_dir.glob("*.json"))
    accuracies: List[float] = []
    num_evaluated: List[int] = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                rec = json.load(f)
            # Always use generation history
            if "generation_history" in rec:
                hist = rec["generation_history"]
                for entry in hist:
                    accuracies.append(entry.get("best_test_acc", 0.0))
                    num_evaluated.append(entry.get("num_evaluated", 0))
        except Exception:
            # Skip malformed files
            continue
    return accuracies, num_evaluated


def compute_series(accs: List[float], threshold: float) -> tuple[list[float], list[float]]:
    fixed_pct: List[float] = []
    median_pct: List[float] = []
    fixed = 0
    running: List[float] = []
    for i, a in enumerate(accs, start=1):
        running.append(a)
        if a >= threshold:
            fixed += 1
        fixed_pct.append(100.0 * fixed / i)
        median_pct.append(100.0 * float(np.median(running)))
    return fixed_pct, median_pct


def main():
    parser = argparse.ArgumentParser(description="Plot GP baseline progress: % fixed and median accuracy vs generation evaluations")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with per-program JSON results (from GP baseline)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Accuracy threshold to count as fixed")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the figure (PNG)")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")

    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    accs, evaluated_counts = load_gp_accuracies(results_dir)
    if not accs:
        print(f"No results found in {results_dir}")
        return

    fixed_pct, median_pct = compute_series(accs, args.threshold)
    x = evaluated_counts

    plt.figure(figsize=(8, 5))
    plt.plot(x, fixed_pct, label="% fixed", color="#1f77b4")
    plt.plot(x, median_pct, label="median accuracy (%)", color="#ff7f0e")
    plt.xlabel("Programs evaluated")
    plt.ylabel("Percentage")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if args.title:
        plt.title(args.title)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved figure to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


