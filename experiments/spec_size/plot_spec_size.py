"""
plot_spec_size.py  –  Specification Size Ablation Plots

Plot A — Final performance vs N (spec_size_ablation.pdf):
  X-axis: N (log scale); Y-axis: % bugs fixed (test_accuracy >= threshold)
  One GBPR curve; dashed horizontal lines for GP and BFS baselines.

Plot B — Learning curves at different N (spec_size_curves.pdf):
  X-axis: gradient steps (reconstructed from val_accs.npy + steps_per_epoch)
  Y-axis: mean val accuracy
  One line per N value.

Directory layout expected:
  saved_data/{program}/n_{N}/seed_{S}/{loss_fn}/job_{job_id}/
    test_results.json
    val_accs.npy

Usage:
    python plot_spec_size.py \
        --saved-data-dir saved_data \
        --gp-results-dir ../gp_baseline/results \
        --exhaustive-results-dir ../gp_baseline/exhaustive_results \
        --output-dir plots \
        --threshold 1.0
"""

import json
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


N_VALUES = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 40000]
BATCH_SIZE = 256
VAL_STEP = 10  # val_accs recorded every valStep=10 epochs
THRESHOLDS = [0.99, 0.999, 1.0]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_mutation_orders(script_dir: Path) -> Dict[str, int]:
    """Return mapping job_id -> mutation_order for BUGGY_MODEL entries."""
    mutation_path = script_dir / "../mutation/results/aggregated_mutations.json"
    if not mutation_path.exists():
        print(f"Warning: aggregated_mutations.json not found at {mutation_path}")
        return {}
    df = pd.read_json(mutation_path)
    orders: Dict[str, int] = {}
    for _, row in df.iterrows():
        if row["execution_result"].get("status") == "BUGGY_MODEL":
            orders[row["job_id"]] = row["mutation_order"]
    return orders


def load_spec_size_results(
    saved_data_dir: Path,
    n_values: List[int],
    loss_fn_name: str = "cross_entropy_loss",
) -> Dict[int, List[dict]]:
    """
    For each N, collect all test_results.json files from:
        saved_data_dir/{program}/n_{N}/seed_*/{loss_fn_name}/job_{job_id}/test_results.json

    Injects '_program' (from path) into each record.
    Returns a dict: N -> list of result dicts.
    """
    results_by_n: Dict[int, List[dict]] = {n: [] for n in n_values}

    for n in n_values:
        pattern = f"*/n_{n}/seed_*/{loss_fn_name}/*/test_results.json"
        for result_file in sorted(saved_data_dir.glob(pattern)):
            try:
                with open(result_file) as f:
                    rec = json.load(f)
                # Inject program name from path (first component relative to saved_data_dir)
                rec["_program"] = result_file.relative_to(saved_data_dir).parts[0]
                results_by_n[n].append(rec)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not read {result_file}: {e}")

    return results_by_n


def load_val_accs_by_n(
    saved_data_dir: Path,
    n_values: List[int],
    loss_fn_name: str = "cross_entropy_loss",
) -> Dict[int, List[np.ndarray]]:
    """
    For each N, collect val_accs.npy arrays from all jobs.

    Returns a dict: N -> list of val_accs arrays (one per job).
    """
    val_accs_by_n: Dict[int, List[np.ndarray]] = {n: [] for n in n_values}

    for n in n_values:
        pattern = f"*/n_{n}/seed_*/{loss_fn_name}/*/val_accs.npy"
        for val_file in sorted(saved_data_dir.glob(pattern)):
            try:
                arr = np.load(val_file)
                val_accs_by_n[n].append(arr)
            except (OSError, ValueError) as e:
                print(f"Warning: could not read {val_file}: {e}")

    return val_accs_by_n


def compute_pct_fixed(results: List[dict], threshold: float = 1.0) -> Optional[float]:
    """Return % of results where test_accuracy >= threshold, or None if empty."""
    if not results:
        return None
    fixed = sum(1 for r in results if r.get("test_accuracy", 0.0) >= threshold)
    return 100.0 * fixed / len(results)


def filter_results_by_n(
    results_by_n: Dict[int, List[dict]],
    n_values: List[int],
    program: Optional[str] = None,
    mutation_order: Optional[int] = None,
    mutation_orders: Optional[Dict[str, int]] = None,
) -> Dict[int, List[dict]]:
    """Return a filtered copy of results_by_n restricted to the given program and/or mutation order."""
    filtered: Dict[int, List[dict]] = {n: [] for n in n_values}
    for n in n_values:
        for rec in results_by_n[n]:
            if program is not None and rec.get("_program") != program:
                continue
            if mutation_order is not None:
                job_id = rec.get("job_id")
                if mutation_orders is None or mutation_orders.get(job_id) != mutation_order:
                    continue
            filtered[n].append(rec)
    return filtered


# ---------------------------------------------------------------------------
# GP / BFS final-performance loading
# ---------------------------------------------------------------------------

def load_gp_final_pct_fixed(results_dir: Path, threshold: float) -> Optional[float]:
    """
    Read all *.json files in results_dir.  Each file has a "program_history"
    list of {step, best_test_acc} entries.  Use the last entry's best_test_acc
    as the final accuracy for that job.

    Returns % of jobs where final best_test_acc >= threshold.
    """
    if not results_dir.exists():
        return None
    files = sorted(results_dir.glob("*.json"))
    finals: List[float] = []
    for fp in files:
        try:
            with open(fp) as f:
                rec = json.load(f)
            hist = rec.get("program_history", [])
            if hist:
                best = max(
                    entry.get("best_test_acc", 0.0)
                    for entry in hist
                    if entry.get("best_test_acc") is not None
                )
                finals.append(float(best))
        except Exception:
            continue
    if not finals:
        return None
    fixed = sum(1 for v in finals if v >= threshold)
    return 100.0 * fixed / len(finals)


# ---------------------------------------------------------------------------
# Plot A — Final performance vs N
# ---------------------------------------------------------------------------

def plot_spec_size_ablation(
    results_by_n: Dict[int, List[dict]],
    threshold: float,
    n_values: List[int],
    gp_pct: Optional[float],
    bfs_pct: Optional[float],
    output_dir: Path,
    filename_suffix: str = "",
    title_suffix: str = "",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_x: List[int] = []
    overall_y: List[float] = []
    for n in n_values:
        pct = compute_pct_fixed(results_by_n[n], threshold)
        if pct is not None:
            overall_x.append(n)
            overall_y.append(pct)

    if not overall_x:
        print(f"No results found for ablation plot{title_suffix}; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    total_results = sum(len(results_by_n[n]) for n in n_values if results_by_n[n])
    ax.plot(
        overall_x,
        overall_y,
        marker="o",
        label=f"GBPR (n={total_results} total)",
        color="#2ca02c",
        linewidth=2,
        markersize=7,
    )

    if gp_pct is not None:
        ax.axhline(
            gp_pct,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.5,
            label=f"GP final ({gp_pct:.1f}%)",
        )
    if bfs_pct is not None:
        ax.axhline(
            bfs_pct,
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.5,
            label=f"BFS final ({bfs_pct:.1f}%)",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of training samples (N)", fontsize=12)
    ax.set_ylabel("% mutations fixed", fontsize=12)
    title = f"Specification Size Ablation (threshold={threshold})"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_xticks(overall_x)
    ax.set_xticklabels([str(n) for n in overall_x], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    thresh_str = f"{threshold:.3f}".rstrip("0").rstrip(".")
    out_path = output_dir / f"spec_size_ablation_thresh{thresh_str}{filename_suffix}.pdf"
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot B — Learning curves at different N
# ---------------------------------------------------------------------------

def plot_learning_curves(
    val_accs_by_n: Dict[int, List[np.ndarray]],
    n_values: List[int],
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    val_step: int = VAL_STEP,
):
    """
    Plot mean val accuracy vs gradient steps for each N value.

    The step axis is reconstructed as:
        step = epoch_idx * val_step * steps_per_epoch
    where steps_per_epoch = max(1, N // batch_size).

    val_accs.npy records one value every val_step epochs, so
    epoch_idx=0 corresponds to epoch 0, epoch_idx=1 to epoch val_step, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    has_data = any(len(val_accs_by_n[n]) > 0 for n in n_values)
    if not has_data:
        print("No val_accs.npy files found for Plot B; skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_values)))

    for n, color in zip(n_values, colors):
        arrays = val_accs_by_n[n]
        if not arrays:
            continue

        steps_per_epoch = max(1, n // batch_size)

        # Align arrays to the shortest length so we can average
        min_len = min(len(a) for a in arrays)
        if min_len == 0:
            continue
        stacked = np.stack([a[:min_len] for a in arrays], axis=0)  # (n_jobs, T)
        mean_acc = np.mean(stacked, axis=0)                          # (T,)

        # Reconstruct gradient-step axis
        # epoch_idx i corresponds to epoch i * val_step
        # steps at that epoch = i * val_step * steps_per_epoch
        steps = np.arange(len(mean_acc)) * val_step * steps_per_epoch

        ax.plot(
            steps,
            mean_acc * 100,
            label=f"N={n} ({len(arrays)} jobs)",
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlabel("Gradient steps", fontsize=12)
    ax.set_ylabel("Mean val accuracy (%)", fontsize=12)
    ax.set_title("Learning Curves by Specification Size", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    out_path = output_dir / "spec_size_curves.pdf"
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--saved-data-dir",
    default="saved_data",
    show_default=True,
    help="Root directory of spec_size saved data",
)
@click.option(
    "--gp-results-dir",
    default=None,
    help="Directory with GP baseline per-job JSON results (for reference line)",
)
@click.option(
    "--exhaustive-results-dir",
    default=None,
    help="Directory with BFS/exhaustive per-job JSON results (for reference line)",
)
@click.option(
    "--output-dir",
    default="plots",
    show_default=True,
    help="Directory to save output plots",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    show_default=True,
    help="Accuracy threshold to count a mutation as fixed (overrides default multi-threshold sweep)",
)
@click.option(
    "--loss-fn-name",
    default="cross_entropy_loss",
    show_default=True,
    help="Loss function subdirectory name",
)
def main(
    saved_data_dir,
    gp_results_dir,
    exhaustive_results_dir,
    output_dir,
    threshold,
    loss_fn_name,
):
    """Plot GBPR % fixed vs. number of training samples (specification size ablation)."""
    script_dir = Path(__file__).parent
    saved_data_path = Path(saved_data_dir) if Path(saved_data_dir).is_absolute() else script_dir / saved_data_dir
    output_path = Path(output_dir) if Path(output_dir).is_absolute() else script_dir / output_dir

    print(f"Loading results from {saved_data_path} ...")
    results_by_n = load_spec_size_results(saved_data_path, N_VALUES, loss_fn_name)
    for n in N_VALUES:
        print(f"  N={n:>6}: {len(results_by_n[n])} results")

    # Load val_accs for Plot B
    val_accs_by_n = load_val_accs_by_n(saved_data_path, N_VALUES, loss_fn_name)
    for n in N_VALUES:
        print(f"  N={n:>6}: {len(val_accs_by_n[n])} val_accs arrays")

    # Load mutation orders for breakdown plots
    mutation_orders = load_mutation_orders(script_dir)
    print(f"Loaded mutation orders for {len(mutation_orders)} jobs")

    # Collect unique programs and mutation order values
    all_programs: List[str] = sorted({
        rec["_program"]
        for recs in results_by_n.values()
        for rec in recs
        if "_program" in rec
    })
    all_orders: List[int] = sorted({
        mutation_orders[rec.get("job_id")]
        for recs in results_by_n.values()
        for rec in recs
        if rec.get("job_id") in mutation_orders
    })
    print(f"Programs: {all_programs}")
    print(f"Mutation orders: {all_orders}")

    # Load GP and BFS final performance (threshold-dependent, cached per threshold)
    def get_gp_pct(thr: float) -> Optional[float]:
        if not gp_results_dir:
            return None
        gp_path = Path(gp_results_dir) if Path(gp_results_dir).is_absolute() else script_dir / gp_results_dir
        pct = load_gp_final_pct_fixed(gp_path, thr)
        if pct is not None:
            print(f"GP final % fixed (thr={thr}): {pct:.1f}%")
        else:
            print(f"Warning: no GP results loaded from {gp_path}")
        return pct

    def get_bfs_pct(thr: float) -> Optional[float]:
        if not exhaustive_results_dir:
            return None
        bfs_path = Path(exhaustive_results_dir) if Path(exhaustive_results_dir).is_absolute() else script_dir / exhaustive_results_dir
        pct = load_gp_final_pct_fixed(bfs_path, thr)
        if pct is not None:
            print(f"BFS final % fixed (thr={thr}): {pct:.1f}%")
        else:
            print(f"Warning: no BFS results loaded from {bfs_path}")
        return pct

    thresholds_to_run = [threshold] if threshold is not None else THRESHOLDS

    for thr in thresholds_to_run:
        gp_pct = get_gp_pct(thr)
        bfs_pct = get_bfs_pct(thr)

        # --- Overall ---
        plot_spec_size_ablation(
            results_by_n=results_by_n,
            threshold=thr,
            n_values=N_VALUES,
            gp_pct=gp_pct,
            bfs_pct=bfs_pct,
            output_dir=output_path,
        )

        # --- Per program ---
        for prog in all_programs:
            filtered = filter_results_by_n(results_by_n, N_VALUES, program=prog)
            safe_prog = prog.replace("/", "_")
            plot_spec_size_ablation(
                results_by_n=filtered,
                threshold=thr,
                n_values=N_VALUES,
                gp_pct=gp_pct,
                bfs_pct=bfs_pct,
                output_dir=output_path,
                filename_suffix=f"_program_{safe_prog}",
                title_suffix=f"Program: {prog}",
            )

        # --- Per mutation order ---
        for order in all_orders:
            filtered = filter_results_by_n(
                results_by_n, N_VALUES, mutation_order=order, mutation_orders=mutation_orders
            )
            plot_spec_size_ablation(
                results_by_n=filtered,
                threshold=thr,
                n_values=N_VALUES,
                gp_pct=gp_pct,
                bfs_pct=bfs_pct,
                output_dir=output_path,
                filename_suffix=f"_order_{order}",
                title_suffix=f"Mutation Order: {order}",
            )

        # --- Per program and mutation order ---
        for prog in all_programs:
            for order in all_orders:
                filtered = filter_results_by_n(
                    results_by_n, N_VALUES,
                    program=prog, mutation_order=order, mutation_orders=mutation_orders,
                )
                # Skip if no data at any N
                if not any(filtered[n] for n in N_VALUES):
                    continue
                safe_prog = prog.replace("/", "_")
                plot_spec_size_ablation(
                    results_by_n=filtered,
                    threshold=thr,
                    n_values=N_VALUES,
                    gp_pct=gp_pct,
                    bfs_pct=bfs_pct,
                    output_dir=output_path,
                    filename_suffix=f"_program_{safe_prog}_order_{order}",
                    title_suffix=f"Program: {prog}, Mutation Order: {order}",
                )

    # Plot B — learning curves at different N (threshold-independent)
    plot_learning_curves(
        val_accs_by_n=val_accs_by_n,
        n_values=N_VALUES,
        output_dir=output_path,
    )


if __name__ == "__main__":
    main()
