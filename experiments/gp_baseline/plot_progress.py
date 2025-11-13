import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_gp_histories(results_dir: Path) -> tuple[Dict[Tuple[str, str], List[Tuple[int, float]]], List[Tuple[str, str]]]:
    """
    Load per-job GP histories as a list of (num_programs, best_test_acc) tuples.
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
            hist = rec.get("program_history", [])
            if not hist:
                continue
            series: List[Tuple[int, float]] = []
            for entry in hist:
                steps = int(entry.get("step", 0))
                best = entry.get("best_test_acc")
                if best is not None and steps > 0:
                    series.append((steps, float(best)))
            series.sort(key=lambda x: x[0])
            if series:
                job_histories[job_key] = series
        except Exception:
            continue
    return job_histories, jobs


def load_exhaustive_histories(results_dir: Path) -> tuple[Dict[Tuple[str, str], List[Tuple[int, float]]], List[Tuple[str, str]]]:
    """
    Load per-job exhaustive search histories as a list of (num_programs, best_test_acc) tuples.
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
            hist = rec.get("program_history", [])
            if not hist:
                continue
            series: List[Tuple[int, float]] = []
            for entry in hist:
                steps = int(entry.get("step", 0))
                best = entry.get("best_test_acc")
                if best is not None and steps > 0:
                    series.append((steps, float(best)))
            series.sort(key=lambda x: x[0])
            if series:
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


def aggregate_job_series(job_to_series: Dict[Tuple[str, str], List[float]], threshold: float) -> tuple[List[int], List[float], List[float], List[float]]:
    """
    Given per-job best-so-far accuracy series (per step), compute aggregated
    median accuracy (%), mean accuracy (%), and % fixed across jobs at each global step.
    Returns (x_steps, fixed_pct, median_pct, mean_pct).
    """
    if not job_to_series:
        return [], [], [], []
    max_len = max(len(s) for s in job_to_series.values())
    x_steps = list(range(1, max_len + 1))
    fixed_pct: List[float] = []
    median_pct: List[float] = []
    mean_pct: List[float] = []
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
        mean_pct.append(100.0 * float(np.mean(vals_np)))
    return x_steps, fixed_pct, median_pct, mean_pct


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


def load_mutation_orders() -> Dict[str, int]:
    """
    Load mutation orders from aggregated mutations file.
    Returns a mapping from job_id to mutation_order.
    """
    try:
        script_dir = Path(__file__).parent
        mutation_path = script_dir.parent / "mutation/results/aggregated_mutations.json"
        if not mutation_path.exists():
            return {}
        df = pd.read_json(mutation_path)
        mutation_orders = {}
        for _, row in df.iterrows():
            if row["execution_result"].get("status") == "BUGGY_MODEL":
                mutation_orders[row["job_id"]] = row["mutation_order"]
        return mutation_orders
    except Exception:
        return {}


def create_plot(
    gp_job_series: Dict[Tuple[str, str], List[float]],
    grad_job_series: Optional[Dict[Tuple[str, str], List[float]]],
    exhaustive_job_series: Optional[Dict[Tuple[str, str], List[float]]],
    threshold: float,
    title: Optional[str] = None,
) -> None:
    """
    Create a plot from the given job series data.
    """
    n_gp = len(gp_job_series)
    x_gp, gp_fixed_pct, gp_median_pct, gp_mean_pct = aggregate_job_series(gp_job_series, threshold)
    
    plt.figure(figsize=(8, 5))
    # GP baseline lines
    plt.plot(x_gp, gp_fixed_pct, label=f"GP % fixed ({n_gp} programs)", color="#1f77b4")
    plt.plot(x_gp, gp_median_pct, label=f"GP median accuracy (%) ({n_gp} programs)", color="#1f77b4", linestyle="--")
    plt.plot(x_gp, gp_mean_pct, label=f"GP avg accuracy (%) ({n_gp} programs)", color="#1f77b4", linestyle=":")
    # Gradient-based lines (if available)
    if grad_job_series:
        n_grad = len(grad_job_series)
        x_grad, grad_fixed_pct, grad_median_pct, grad_mean_pct = aggregate_job_series(grad_job_series, threshold)
        plt.plot(x_grad, grad_fixed_pct, label=f"Grad % fixed ({n_grad} programs)", color="#2ca02c")
        plt.plot(x_grad, grad_median_pct, label=f"Grad median accuracy (%) ({n_grad} programs)", color="#2ca02c", linestyle="--")
        plt.plot(x_grad, grad_mean_pct, label=f"Grad avg accuracy (%) ({n_grad} programs)", color="#2ca02c", linestyle=":")
    # Exhaustive search lines (if available)
    if exhaustive_job_series:
        n_exh = len(exhaustive_job_series)
        x_exh, exh_fixed_pct, exh_median_pct, exh_mean_pct = aggregate_job_series(exhaustive_job_series, threshold)
        plt.plot(x_exh, exh_fixed_pct, label=f"Exhaustive % fixed ({n_exh} programs)", color="#ff7f0e")
        plt.plot(x_exh, exh_median_pct, label=f"Exhaustive median accuracy (%) ({n_exh} programs)", color="#ff7f0e", linestyle="--")
        plt.plot(x_exh, exh_mean_pct, label=f"Exhaustive avg accuracy (%) ({n_exh} programs)", color="#ff7f0e", linestyle=":")
    plt.xlabel("Programs generated")
    plt.ylabel("Percentage")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if title:
        plt.title(title)


def main():
    parser = argparse.ArgumentParser(description="Plot GP baseline progress: % fixed and median accuracy vs generation evaluations")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with per-program JSON results (from GP baseline)")
    parser.add_argument("--exhaustive-results-dir", type=str, default=None, help="Directory with per-program JSON results (from exhaustive search)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Accuracy threshold to count as fixed")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save plots (PDF format)")
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

    # Optionally load gradient-based series
    grad_job_series: Dict[Tuple[str, str], List[float]] = {}
    if args.saved_data_dir:
        saved_data_dir = Path(args.saved_data_dir).resolve()
        grad_job_series = load_gradient_job_series(saved_data_dir, args.loss_function, jobs)

    # Optionally load exhaustive search series
    exhaustive_job_series: Dict[Tuple[str, str], List[float]] = {}
    if args.exhaustive_results_dir:
        exhaustive_results_dir = Path(args.exhaustive_results_dir).resolve()
        exhaustive_histories, exhaustive_jobs = load_exhaustive_histories(exhaustive_results_dir)
        for job_key, hist in exhaustive_histories.items():
            if job_key in gp_job_series:
                exhaustive_job_series[job_key] = expand_gp_history_to_series(hist)

    # Find intersection of all available series (only include jobs present in all methods)
    all_job_sets = [set(gp_job_series.keys())]
    if grad_job_series:
        all_job_sets.append(set(grad_job_series.keys()))
    if exhaustive_job_series:
        all_job_sets.append(set(exhaustive_job_series.keys()))
    
    # Only compute intersection if we have at least GP jobs
    if all_job_sets:
        common_jobs = set.intersection(*all_job_sets)
    else:
        common_jobs = set()
    
    # Filter all series to only include common jobs
    gp_job_series = {job: gp_job_series[job] for job in common_jobs}
    if grad_job_series:
        grad_job_series = {job: grad_job_series[job] for job in common_jobs}
    if exhaustive_job_series:
        exhaustive_job_series = {job: exhaustive_job_series[job] for job in common_jobs}
    
    # Update jobs list to only include common jobs
    jobs = list(common_jobs)

    # Load mutation orders
    mutation_orders = load_mutation_orders()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot all data combined (original plot)
    create_plot(
        gp_job_series,
        grad_job_series if grad_job_series else None,
        exhaustive_job_series if exhaustive_job_series else None,
        args.threshold,
        title="All Programs Combined"
    )
    plt.savefig(output_dir / "all_programs.pdf", bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved figure to {output_dir / 'all_programs.pdf'}")

    # Group jobs by program
    jobs_by_program: Dict[str, List[Tuple[str, str]]] = {}
    for prog, job_id in jobs:
        if prog not in jobs_by_program:
            jobs_by_program[prog] = []
        jobs_by_program[prog].append((prog, job_id))

    # Group jobs by mutation order
    jobs_by_mutation_order: Dict[int, List[Tuple[str, str]]] = {}
    for prog, job_id in jobs:
        order = mutation_orders.get(job_id)
        if order is not None:
            if order not in jobs_by_mutation_order:
                jobs_by_mutation_order[order] = []
            jobs_by_mutation_order[order].append((prog, job_id))

    # Group jobs by program and mutation order
    jobs_by_program_and_order: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
    for prog, job_id in jobs:
        order = mutation_orders.get(job_id)
        if order is not None:
            key = (prog, order)
            if key not in jobs_by_program_and_order:
                jobs_by_program_and_order[key] = []
            jobs_by_program_and_order[key].append((prog, job_id))

    # 2. Plots per program
    for prog, prog_jobs in jobs_by_program.items():
        prog_gp_series = {job: gp_job_series[job] for job in prog_jobs if job in gp_job_series}
        prog_grad_series = {job: grad_job_series[job] for job in prog_jobs if job in grad_job_series} if grad_job_series else None
        prog_exh_series = {job: exhaustive_job_series[job] for job in prog_jobs if job in exhaustive_job_series} if exhaustive_job_series else None
        if prog_gp_series:
            create_plot(prog_gp_series, prog_grad_series, prog_exh_series, args.threshold, title=f"Program: {prog}")
            safe_prog_name = prog.replace("/", "_")
            plt.savefig(output_dir / f"program_{safe_prog_name}.pdf", bbox_inches="tight", format="pdf")
            plt.close()
            print(f"Saved figure to {output_dir / f'program_{safe_prog_name}.pdf'}")

    # 3. Plots per mutation order (all programs combined)
    for order in sorted(jobs_by_mutation_order.keys()):
        order_jobs = jobs_by_mutation_order[order]
        order_gp_series = {job: gp_job_series[job] for job in order_jobs if job in gp_job_series}
        order_grad_series = {job: grad_job_series[job] for job in order_jobs if job in grad_job_series} if grad_job_series else None
        order_exh_series = {job: exhaustive_job_series[job] for job in order_jobs if job in exhaustive_job_series} if exhaustive_job_series else None
        if order_gp_series:
            create_plot(order_gp_series, order_grad_series, order_exh_series, args.threshold, title=f"Mutation Order: {order} (All Programs)")
            plt.savefig(output_dir / f"mutation_order_{order}.pdf", bbox_inches="tight", format="pdf")
            plt.close()
            print(f"Saved figure to {output_dir / f'mutation_order_{order}.pdf'}")

    # 4. Plots per program and mutation order
    for (prog, order), prog_order_jobs in sorted(jobs_by_program_and_order.items()):
        prog_order_gp_series = {job: gp_job_series[job] for job in prog_order_jobs if job in gp_job_series}
        prog_order_grad_series = {job: grad_job_series[job] for job in prog_order_jobs if job in grad_job_series} if grad_job_series else None
        prog_order_exh_series = {job: exhaustive_job_series[job] for job in prog_order_jobs if job in exhaustive_job_series} if exhaustive_job_series else None
        if prog_order_gp_series:
            create_plot(prog_order_gp_series, prog_order_grad_series, prog_order_exh_series, args.threshold, title=f"Program: {prog}, Mutation Order: {order}")
            safe_prog_name = prog.replace("/", "_")
            plt.savefig(output_dir / f"program_{safe_prog_name}_order_{order}.pdf", bbox_inches="tight", format="pdf")
            plt.close()
            print(f"Saved figure to {output_dir / f'program_{safe_prog_name}_order_{order}.pdf'}")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()


