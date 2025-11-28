import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_gp_histories(
    results_dir: Path,
) -> tuple[Dict[Tuple[str, str], List[Tuple[int, float]]], List[Tuple[str, str]]]:
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


def load_exhaustive_histories(
    results_dir: Path,
) -> tuple[Dict[Tuple[str, str], List[Tuple[int, float]]], List[Tuple[str, str]]]:
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


def load_gradient_job_series(
    saved_data_dir: Path, loss_fn: str, jobs: List[Tuple[str, str]]
) -> Dict[Tuple[str, str], List[float]]:
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


def aggregate_job_series(
    job_to_series: Dict[Tuple[str, str], List[float]], threshold: float
) -> tuple[List[int], List[float], List[float], List[float]]:
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
    show_median_avg: bool = False,
) -> None:
    """
    Create a plot from the given job series data.
    
    Args:
        show_median_avg: If True, show median and average accuracy lines. Default False.
    """
    n_gp = len(gp_job_series)
    x_gp, gp_fixed_pct, gp_median_pct, gp_mean_pct = aggregate_job_series(
        gp_job_series, threshold
    )

    target_epoch = 500
    plt.figure(figsize=(8, 5))
    # GP baseline lines
    plt.plot(x_gp, gp_fixed_pct, label="GP", color="#1f77b4")
    if x_gp and x_gp[-1] < target_epoch:
        plt.plot(
            [x_gp[-1], target_epoch],
            [gp_fixed_pct[-1], gp_fixed_pct[-1]],
            color="#1f77b4",
            linestyle="--",
            dashes=(1, 1),
        )

    if show_median_avg:
        plt.plot(
            x_gp,
            gp_median_pct,
            label="GP median accuracy (%)",
            color="#1f77b4",
            linestyle="--",
        )
        if x_gp and x_gp[-1] < target_epoch:
            plt.plot(
                [x_gp[-1], target_epoch],
                [gp_median_pct[-1], gp_median_pct[-1]],
                color="#1f77b4",
                linestyle="--",
                dashes=(1, 1),
            )
        plt.plot(
            x_gp,
            gp_mean_pct,
            label="GP avg accuracy (%)",
            color="#1f77b4",
            linestyle=":",
        )
        if x_gp and x_gp[-1] < target_epoch:
            plt.plot(
                [x_gp[-1], target_epoch],
                [gp_mean_pct[-1], gp_mean_pct[-1]],
                color="#1f77b4",
                linestyle="--",
                dashes=(1, 1),
            )
    # Gradient-based lines (if available)
    if grad_job_series:
        x_grad, grad_fixed_pct, grad_median_pct, grad_mean_pct = aggregate_job_series(
            grad_job_series, threshold
        )
        plt.plot(
            x_grad,
            grad_fixed_pct,
            label="GBPR",
            color="#2ca02c",
        )
        if x_grad and x_grad[-1] < target_epoch:
            plt.plot(
                [x_grad[-1], target_epoch],
                [grad_fixed_pct[-1], grad_fixed_pct[-1]],
                color="#2ca02c",
                linestyle="--",
                dashes=(1, 1),
            )
        if show_median_avg:
            plt.plot(
                x_grad,
                grad_median_pct,
                label="GBPR median accuracy (%)",
                color="#2ca02c",
                linestyle="--",
            )
            if x_grad and x_grad[-1] < target_epoch:
                plt.plot(
                    [x_grad[-1], target_epoch],
                    [grad_median_pct[-1], grad_median_pct[-1]],
                    color="#2ca02c",
                    linestyle="--",
                    dashes=(1, 1),
                )
            plt.plot(
                x_grad,
                grad_mean_pct,
                label="GBPR avg accuracy (%)",
                color="#2ca02c",
                linestyle=":",
            )
            if x_grad and x_grad[-1] < target_epoch:
                plt.plot(
                    [x_grad[-1], target_epoch],
                    [grad_mean_pct[-1], grad_mean_pct[-1]],
                    color="#2ca02c",
                    linestyle="--",
                    dashes=(1, 1),
                )
    # Exhaustive search lines (if available)
    if exhaustive_job_series:
        x_exh, exh_fixed_pct, exh_median_pct, exh_mean_pct = aggregate_job_series(
            exhaustive_job_series, threshold
        )
        plt.plot(
            x_exh,
            exh_fixed_pct,
            label="BFS",
            color="#ff7f0e",
        )
        if x_exh and x_exh[-1] < target_epoch:
            plt.plot(
                [x_exh[-1], target_epoch],
                [exh_fixed_pct[-1], exh_fixed_pct[-1]],
                color="#ff7f0e",
                linestyle="--",
                dashes=(1, 1),
            )
        if show_median_avg:
            plt.plot(
                x_exh,
                exh_median_pct,
                label="BFS median accuracy (%)",
                color="#ff7f0e",
                linestyle="--",
            )
            if x_exh and x_exh[-1] < target_epoch:
                plt.plot(
                    [x_exh[-1], target_epoch],
                    [exh_median_pct[-1], exh_median_pct[-1]],
                    color="#ff7f0e",
                    linestyle="--",
                    dashes=(1, 1),
                )
            plt.plot(
                x_exh,
                exh_mean_pct,
                label="BFS avg accuracy (%)",
                color="#ff7f0e",
                linestyle=":",
            )
            if x_exh and x_exh[-1] < target_epoch:
                plt.plot(
                    [x_exh[-1], target_epoch],
                    [exh_mean_pct[-1], exh_mean_pct[-1]],
                    color="#ff7f0e",
                    linestyle="--",
                    dashes=(1, 1),
                )
    plt.xlabel("Epochs")
    plt.ylabel("% fixed programs")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Add program count to title
    if title:
        plt.title(f"{title} ({n_gp} programs)")
    else:
        plt.title(f"({n_gp} programs)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot GP baseline progress: % fixed and median accuracy vs generation evaluations"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory with per-program JSON results (from GP baseline)",
    )
    parser.add_argument(
        "--exhaustive-results-dir",
        type=str,
        default=None,
        help="Directory with per-program JSON results (from exhaustive search)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Accuracy threshold to count as fixed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots (PDF format)",
    )
    parser.add_argument(
        "--saved-data-dir",
        type=str,
        default=None,
        help="Path to train_mutations saved_data directory to overlay gradient-based progress",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="cross_entropy_loss",
        help="Loss function subdirectory name under saved_data (e.g., cross_entropy_loss)",
    )
    parser.add_argument(
        "--show-median-avg",
        action="store_true",
        default=False,
        help="Show median and average accuracy lines in plots (default: False)",
    )

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
        grad_job_series = load_gradient_job_series(
            saved_data_dir, args.loss_function, jobs
        )

    # Optionally load exhaustive search series
    exhaustive_job_series: Dict[Tuple[str, str], List[float]] = {}
    if args.exhaustive_results_dir:
        exhaustive_results_dir = Path(args.exhaustive_results_dir).resolve()
        exhaustive_histories, exhaustive_jobs = load_exhaustive_histories(
            exhaustive_results_dir
        )
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
        title="All Programs Combined",
        show_median_avg=args.show_median_avg,
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

    # Find the "most_freq" program (by name, not by frequency)
    most_freq_program = None
    for prog in jobs_by_program.keys():
        if prog == "most_freq" or prog == "most-freq":
            most_freq_program = prog
            break

    if most_freq_program:
        print(
            f"Excluding program: {most_freq_program} ({len(jobs_by_program[most_freq_program])} jobs)"
        )
    else:
        print("Warning: 'most_freq' or 'most-freq' program not found in data")

    # Create filtered job series excluding the "most_freq" program
    filtered_jobs = (
        [job for job in jobs if job[0] != most_freq_program]
        if most_freq_program
        else jobs
    )
    filtered_gp_job_series = {
        job: gp_job_series[job] for job in filtered_jobs if job in gp_job_series
    }
    filtered_grad_job_series = (
        {job: grad_job_series[job] for job in filtered_jobs if job in grad_job_series}
        if grad_job_series
        else None
    )
    filtered_exh_job_series = (
        {
            job: exhaustive_job_series[job]
            for job in filtered_jobs
            if job in exhaustive_job_series
        }
        if exhaustive_job_series
        else None
    )

    # 1b. Plot aggregate excluding "most_freq" program
    if most_freq_program and filtered_gp_job_series:
        create_plot(
            filtered_gp_job_series,
            filtered_grad_job_series if filtered_grad_job_series else None,
            filtered_exh_job_series if filtered_exh_job_series else None,
            args.threshold,
            title=f"All Programs Combined (excluding {most_freq_program})",
            show_median_avg=args.show_median_avg,
        )
        safe_prog_name = most_freq_program.replace("/", "_")
        plt.savefig(
            output_dir / f"all_programs_excluding_{safe_prog_name}.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close()
        print(
            f"Saved figure to {output_dir / f'all_programs_excluding_{safe_prog_name}.pdf'}"
        )

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
        prog_gp_series = {
            job: gp_job_series[job] for job in prog_jobs if job in gp_job_series
        }
        prog_grad_series = (
            {job: grad_job_series[job] for job in prog_jobs if job in grad_job_series}
            if grad_job_series
            else None
        )
        prog_exh_series = (
            {
                job: exhaustive_job_series[job]
                for job in prog_jobs
                if job in exhaustive_job_series
            }
            if exhaustive_job_series
            else None
        )
        if prog_gp_series:
            create_plot(
                prog_gp_series,
                prog_grad_series,
                prog_exh_series,
                args.threshold,
                title=f"Program: {prog}",
                show_median_avg=args.show_median_avg,
            )
            safe_prog_name = prog.replace("/", "_")
            plt.savefig(
                output_dir / f"program_{safe_prog_name}.pdf",
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()
            print(f"Saved figure to {output_dir / f'program_{safe_prog_name}.pdf'}")

    # 3. Plots per mutation order (all programs combined)
    for order in sorted(jobs_by_mutation_order.keys()):
        order_jobs = jobs_by_mutation_order[order]
        order_gp_series = {
            job: gp_job_series[job] for job in order_jobs if job in gp_job_series
        }
        order_grad_series = (
            {job: grad_job_series[job] for job in order_jobs if job in grad_job_series}
            if grad_job_series
            else None
        )
        order_exh_series = (
            {
                job: exhaustive_job_series[job]
                for job in order_jobs
                if job in exhaustive_job_series
            }
            if exhaustive_job_series
            else None
        )
        if order_gp_series:
            create_plot(
                order_gp_series,
                order_grad_series,
                order_exh_series,
                args.threshold,
                title=f"Mutation Order: {order}",
                show_median_avg=args.show_median_avg,
            )
            plt.savefig(
                output_dir / f"mutation_order_{order}.pdf",
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()
            print(f"Saved figure to {output_dir / f'mutation_order_{order}.pdf'}")

    # 3b. Plots per mutation order excluding "most_freq" program
    jobs_by_mutation_order_filtered: Dict[int, List[Tuple[str, str]]] = {}
    if most_freq_program:
        for prog, job_id in filtered_jobs:
            order = mutation_orders.get(job_id)
            if order is not None:
                if order not in jobs_by_mutation_order_filtered:
                    jobs_by_mutation_order_filtered[order] = []
                jobs_by_mutation_order_filtered[order].append((prog, job_id))

        for order in sorted(jobs_by_mutation_order_filtered.keys()):
            order_jobs = jobs_by_mutation_order_filtered[order]
            order_gp_series = {
                job: filtered_gp_job_series[job]
                for job in order_jobs
                if job in filtered_gp_job_series
            }
            order_grad_series = (
                {
                    job: filtered_grad_job_series[job]
                    for job in order_jobs
                    if job in filtered_grad_job_series
                }
                if filtered_grad_job_series
                else None
            )
            order_exh_series = (
                {
                    job: filtered_exh_job_series[job]
                    for job in order_jobs
                    if job in filtered_exh_job_series
                }
                if filtered_exh_job_series
                else None
            )
            if order_gp_series:
                safe_prog_name = most_freq_program.replace("/", "_")
                create_plot(
                    order_gp_series,
                    order_grad_series,
                    order_exh_series,
                    args.threshold,
                    title=f"Mutation Order: {order} (All Programs, excluding {most_freq_program})",
                    show_median_avg=args.show_median_avg,
                )
                plt.savefig(
                    output_dir
                    / f"mutation_order_{order}_excluding_{safe_prog_name}.pdf",
                    bbox_inches="tight",
                    format="pdf",
                )
                plt.close()
                print(
                    f"Saved figure to {output_dir / f'mutation_order_{order}_excluding_{safe_prog_name}.pdf'}"
                )

    # 3c. Plots for mutation order >= N (where N in [2, 3, 4])
    for min_order in [2, 3, 4]:
        # Collect all jobs with mutation_order >= min_order
        order_ge_jobs = []
        for prog, job_id in jobs:
            order = mutation_orders.get(job_id)
            if order is not None and order >= min_order:
                order_ge_jobs.append((prog, job_id))

        if order_ge_jobs:
            order_ge_gp_series = {
                job: gp_job_series[job] for job in order_ge_jobs if job in gp_job_series
            }
            order_ge_grad_series = (
                {
                    job: grad_job_series[job]
                    for job in order_ge_jobs
                    if job in grad_job_series
                }
                if grad_job_series
                else None
            )
            order_ge_exh_series = (
                {
                    job: exhaustive_job_series[job]
                    for job in order_ge_jobs
                    if job in exhaustive_job_series
                }
                if exhaustive_job_series
                else None
            )
            if order_ge_gp_series:
                create_plot(
                    order_ge_gp_series,
                    order_ge_grad_series,
                    order_ge_exh_series,
                    args.threshold,
                    title=f"Mutation Order >= {min_order}",
                    show_median_avg=args.show_median_avg,
                )
                plt.savefig(
                    output_dir / f"mutation_order_ge_{min_order}.pdf",
                    bbox_inches="tight",
                    format="pdf",
                )
                plt.close()
                print(
                    f"Saved figure to {output_dir / f'mutation_order_ge_{min_order}.pdf'}"
                )

    # 3d. Plots for mutation order >= N excluding "most_freq" program
    if most_freq_program:
        for min_order in [2, 3, 4]:
            # Collect all filtered jobs with mutation_order >= min_order
            order_ge_filtered_jobs = []
            for prog, job_id in filtered_jobs:
                order = mutation_orders.get(job_id)
                if order is not None and order >= min_order:
                    order_ge_filtered_jobs.append((prog, job_id))

            if order_ge_filtered_jobs:
                order_ge_gp_series = {
                    job: filtered_gp_job_series[job]
                    for job in order_ge_filtered_jobs
                    if job in filtered_gp_job_series
                }
                order_ge_grad_series = (
                    {
                        job: filtered_grad_job_series[job]
                        for job in order_ge_filtered_jobs
                        if job in filtered_grad_job_series
                    }
                    if filtered_grad_job_series
                    else None
                )
                order_ge_exh_series = (
                    {
                        job: filtered_exh_job_series[job]
                        for job in order_ge_filtered_jobs
                        if job in filtered_exh_job_series
                    }
                    if filtered_exh_job_series
                    else None
                )
                if order_ge_gp_series:
                    safe_prog_name = most_freq_program.replace("/", "_")
                    create_plot(
                        order_ge_gp_series,
                        order_ge_grad_series,
                        order_ge_exh_series,
                        args.threshold,
                        title=f"Mutation Order >= {min_order} (All Programs, excluding {most_freq_program})",
                        show_median_avg=args.show_median_avg,
                    )
                    plt.savefig(
                        output_dir
                        / f"mutation_order_ge_{min_order}_excluding_{safe_prog_name}.pdf",
                        bbox_inches="tight",
                        format="pdf",
                    )
                    plt.close()
                    print(
                        f"Saved figure to {output_dir / f'mutation_order_ge_{min_order}_excluding_{safe_prog_name}.pdf'}"
                    )

    # 4. Plots per program and mutation order
    for (prog, order), prog_order_jobs in sorted(jobs_by_program_and_order.items()):
        prog_order_gp_series = {
            job: gp_job_series[job] for job in prog_order_jobs if job in gp_job_series
        }
        prog_order_grad_series = (
            {
                job: grad_job_series[job]
                for job in prog_order_jobs
                if job in grad_job_series
            }
            if grad_job_series
            else None
        )
        prog_order_exh_series = (
            {
                job: exhaustive_job_series[job]
                for job in prog_order_jobs
                if job in exhaustive_job_series
            }
            if exhaustive_job_series
            else None
        )
        if prog_order_gp_series:
            create_plot(
                prog_order_gp_series,
                prog_order_grad_series,
                prog_order_exh_series,
                args.threshold,
                title=f"Program: {prog}, Mutation Order: {order}",
                show_median_avg=args.show_median_avg,
            )
            safe_prog_name = prog.replace("/", "_")
            plt.savefig(
                output_dir / f"program_{safe_prog_name}_order_{order}.pdf",
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()
            print(
                f"Saved figure to {output_dir / f'program_{safe_prog_name}_order_{order}.pdf'}"
            )

    # 5. Plot pass rate by mutation order (with and without most-freq)
    def compute_pass_rate_by_order(
        job_series: Dict[Tuple[str, str], List[float]],
        jobs_by_order: Dict[int, List[Tuple[str, str]]],
        threshold: float,
    ) -> Dict[int, float]:
        """Compute final pass rate for each mutation order."""
        pass_rates = {}
        for order in sorted(jobs_by_order.keys()):
            order_jobs = jobs_by_order[order]
            passed = 0
            total = 0
            for job in order_jobs:
                if job in job_series:
                    series = job_series[job]
                    if series:
                        final_acc = series[-1]
                        if final_acc >= threshold:
                            passed += 1
                        total += 1
            if total > 0:
                pass_rates[order] = 100.0 * passed / total
            else:
                pass_rates[order] = 0.0
        return pass_rates

    def plot_pass_rate_by_mutation_order(
        gp_series: Dict[Tuple[str, str], List[float]],
        grad_series: Optional[Dict[Tuple[str, str], List[float]]],
        exh_series: Optional[Dict[Tuple[str, str], List[float]]],
        jobs_by_order: Dict[int, List[Tuple[str, str]]],
        threshold: float,
        title_suffix: str,
        filename_base: str,
        output_dir: Path,
    ) -> Dict:
        """Create plots showing pass rate by mutation order for GP, GBPR, and BFS.
        Returns the data dictionary for JSON output."""
        # Compute pass rates for each method
        gp_rates = compute_pass_rate_by_order(gp_series, jobs_by_order, threshold)
        
        grad_rates = {}
        if grad_series:
            grad_rates = compute_pass_rate_by_order(grad_series, jobs_by_order, threshold)
        
        exh_rates = {}
        if exh_series:
            exh_rates = compute_pass_rate_by_order(exh_series, jobs_by_order, threshold)
        
        # Get all mutation orders
        all_orders = sorted(set(list(gp_rates.keys()) + list(grad_rates.keys()) + list(exh_rates.keys())))
        if not all_orders:
            return {}
        
        # Prepare data for plotting
        gp_values = [gp_rates.get(order, 0.0) for order in all_orders]
        grad_values = [grad_rates.get(order, 0.0) for order in all_orders] if grad_rates else []
        exh_values = [exh_rates.get(order, 0.0) for order in all_orders] if exh_rates else []
        
        # Prepare data dictionary for JSON
        data_dict = {
            "mutation_orders": all_orders,
            "gp": {order: gp_rates.get(order, 0.0) for order in all_orders},
        }
        if grad_rates:
            data_dict["gbpr"] = {order: grad_rates.get(order, 0.0) for order in all_orders}
        if exh_rates:
            data_dict["bfs"] = {order: exh_rates.get(order, 0.0) for order in all_orders}
        
        # Create grouped bar chart
        x = np.arange(len(all_orders))
        num_methods = sum([1, bool(grad_values), bool(exh_values)])
        width = 0.8 / num_methods if num_methods > 0 else 0.8
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = []
        offset = -width * (num_methods - 1) / 2
        
        # GP bars
        bars.append(ax.bar(x + offset, gp_values, width, label='GP', color='#1f77b4'))
        offset += width
        
        # GBPR bars
        if grad_values:
            bars.append(ax.bar(x + offset, grad_values, width, label='GBPR', color='#2ca02c'))
            offset += width
        
        # BFS (Exhaustive) bars
        if exh_values:
            bars.append(ax.bar(x + offset, exh_values, width, label='BFS', color='#ff7f0e'))
            offset += width
        
        ax.set_xlabel('Mutation Order')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title(f'Pass Rate by Mutation Order{title_suffix}')
        ax.set_xticks(x)
        ax.set_xticklabels(all_orders)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        bar_filename = filename_base.replace('.pdf', '_bars.pdf')
        plt.savefig(output_dir / bar_filename, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"Saved figure to {output_dir / bar_filename}")
        
        # Create line plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(all_orders, gp_values, marker='o', label='GP', color='#1f77b4', linewidth=2, markersize=8)
        if grad_values:
            ax.plot(all_orders, grad_values, marker='s', label='GBPR', color='#2ca02c', linewidth=2, markersize=8)
        if exh_values:
            ax.plot(all_orders, exh_values, marker='^', label='BFS', color='#ff7f0e', linewidth=2, markersize=8)
        
        ax.set_xlabel('Mutation Order')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title(f'Pass Rate by Mutation Order{title_suffix}')
        ax.set_xticks(all_orders)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        line_filename = filename_base.replace('.pdf', '_lines.pdf')
        plt.savefig(output_dir / line_filename, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"Saved figure to {output_dir / line_filename}")
        
        return data_dict

    # Plot 1: With most-freq (all programs)
    all_data = plot_pass_rate_by_mutation_order(
        gp_job_series,
        grad_job_series if grad_job_series else None,
        exhaustive_job_series if exhaustive_job_series else None,
        jobs_by_mutation_order,
        args.threshold,
        "",
        "pass_rate_by_mutation_order_all.pdf",
        output_dir,
    )
    
    # Plot 2: Without most-freq
    filtered_data = {}
    if most_freq_program and jobs_by_mutation_order_filtered:
        filtered_data = plot_pass_rate_by_mutation_order(
            filtered_gp_job_series,
            filtered_grad_job_series if filtered_grad_job_series else None,
            filtered_exh_job_series if filtered_exh_job_series else None,
            jobs_by_mutation_order_filtered,
            args.threshold,
            f" (Excluding {most_freq_program})",
            f"pass_rate_by_mutation_order_excluding_{most_freq_program.replace('/', '_')}.pdf",
            output_dir,
        )

    # Print JSON results to stdout
    results_json = {
        "threshold": args.threshold,
        "all_programs": all_data,
    }
    if filtered_data:
        results_json["excluding_most_freq"] = filtered_data
    
    print("\n" + "="*80)
    print("Pass Rate by Mutation Order Results (JSON):")
    print("="*80)
    print(json.dumps(results_json, indent=2))
    print("="*80)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
