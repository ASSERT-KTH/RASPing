import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_gp_histories(results_dir: Path) -> tuple[Dict[Tuple[str, str], List[Tuple[int, float]]], List[Tuple[str, str]]]:
    """
    Load per-job GP histories as a list of (num_evaluated, best_test_acc) tuples.
    Returns (job_histories, jobs_list)
    """
    files = sorted(results_dir.glob("*.json"))
    job_histories: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
    jobs: List[Tuple[str, str]] = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                rec = json.load(f)
            if "program_name" not in rec or "job_id" not in rec:
                continue
            job_key = (rec["program_name"], rec["job_id"])
            jobs.append(job_key)
            hist = rec.get("generation_history", [])
            if not hist:
                continue
            series: List[Tuple[int, float]] = []
            for entry in hist:
                steps = int(entry.get("num_evaluated", 0))
                best = float(entry.get("best_test_acc", 0.0))
                if steps > 0:
                    series.append((steps, best))
            series.sort(key=lambda x: x[0])
            job_histories[job_key] = series
        except Exception:
            continue
    return job_histories, jobs


def _map_program_for_saved(prog: str) -> str:
    if prog == "most-freq":
        return "most_freq"
    if prog == "shuffle_dyck1":
        return "shuffle_dyck"
    return prog


def load_gradient_job_series(saved_data_dir: Path, loss_fn: str, jobs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[float]]:
    """
    Load gradient-based repair progress as a single flattened series across jobs.
    Uses per-epoch validation accuracy when available; applies cumulative best per job.
    """
    # TODO: We need to store the test results
    job_to_series: Dict[Tuple[str, str], List[float]] = {}
    for prog, job_id in jobs:
        prog_dir = _map_program_for_saved(prog)
        job_dir = saved_data_dir / prog_dir / loss_fn / f"job_{job_id}"
        val_path = job_dir / "val_accs.npy"
        train_path = job_dir / "train_accs.npy"
        arr = None
        try:
            if val_path.exists():
                arr = np.load(val_path)
            elif train_path.exists():
                arr = np.load(train_path)
        except Exception:
            arr = None
        if arr is None:
            continue
        arr = np.asarray(arr).reshape(-1)
        # Best-so-far per epoch within this job
        best = np.maximum.accumulate(arr)
        job_to_series[(prog, job_id)] = best.tolist()
    return job_to_series


def aggregate_job_series(job_to_series: Dict[Tuple[str, str], List[float]], threshold: float) -> tuple[List[int], List[float], List[float]]:
    """
    Given per-job best-so-far accuracy series (per step), compute aggregated
    median accuracy (%) and % fixed across jobs at each global step.
    Returns (x_steps, fixed_pct, median_pct).
    """
    if not job_to_series:
        return [], [], []
    max_len = max(len(s) for s in job_to_series.values())
    x_steps = list(range(1, max_len + 1))
    fixed_pct: List[float] = []
    median_pct: List[float] = []
    for s in x_steps:
        vals = []
        for series in job_to_series.values():
            if not series:
                vals.append(0.0)
            else:
                idx = min(s, len(series)) - 1
                vals.append(float(series[idx]))
        vals_np = np.array(vals, dtype=float)
        fixed_pct.append(100.0 * float(np.mean(vals_np >= threshold)))
        median_pct.append(100.0 * float(np.median(vals_np)))
    return x_steps, fixed_pct, median_pct


def expand_gp_history_to_series(hist: List[Tuple[int, float]]) -> List[float]:
    """
    Convert sparse (steps, best) history to a dense per-step best-so-far series.
    """
    if not hist:
        return []
    last_step = hist[-1][0]
    out: List[float] = []
    ptr = 0
    best = 0.0
    for s in range(1, last_step + 1):
        while ptr < len(hist) and hist[ptr][0] == s:
            best = max(best, hist[ptr][1])
            ptr += 1
        out.append(best)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot GP baseline progress: % fixed and median accuracy vs generation evaluations")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with per-program JSON results (from GP baseline)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Accuracy threshold to count as fixed")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the figure (PNG)")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    parser.add_argument("--saved-data-dir", type=str, default=None, help="Path to train_mutations saved_data directory to overlay gradient-based progress")
    parser.add_argument("--loss-function", type=str, default="cross_entropy_loss", help="Loss function subdirectory name under saved_data (e.g., cross_entropy_loss)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    gp_histories, jobs = load_gp_histories(results_dir)
    if not gp_histories:
        print(f"No results found in {results_dir}")
        return

    # Build per-job dense best-so-far series for GP
    gp_job_series: Dict[Tuple[str, str], List[float]] = {}
    for job_key, hist in gp_histories.items():
        gp_job_series[job_key] = expand_gp_history_to_series(hist)

    x_gp, gp_fixed_pct, gp_median_pct = aggregate_job_series(gp_job_series, args.threshold)

    # Optionally load gradient-based series aligned on the same jobs
    grad_job_series: Dict[Tuple[str, str], List[float]] = {}
    if args.saved_data_dir:
        saved_data_dir = Path(args.saved_data_dir).resolve()
        grad_job_series = load_gradient_job_series(saved_data_dir, args.loss_function, jobs)
        # TODO: This might not correct, we should multiply the step count by the batch size
        x_grad, grad_fixed_pct, grad_median_pct = aggregate_job_series(grad_job_series, args.threshold)
        # x_grad = [x * 156 for x in x_grad]

    plt.figure(figsize=(8, 5))
    # GP baseline lines
    plt.plot(x_gp, gp_fixed_pct, label="GP % fixed", color="#1f77b4")
    plt.plot(x_gp, gp_median_pct, label="GP median accuracy (%)", color="#1f77b4", linestyle="--")
    # Gradient-based lines (if available)
    if grad_job_series:
        plt.plot(x_grad, grad_fixed_pct, label="Grad % fixed", color="#2ca02c")
        plt.plot(x_grad, grad_median_pct, label="Grad median accuracy (%)", color="#2ca02c", linestyle="--")
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


