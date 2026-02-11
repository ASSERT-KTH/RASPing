"""
plot_spec_size.py  –  Specification Size Ablation Plot

For each N in N_VALUES, reads test_results.json from all (program, job_id)
pairs and computes pct_fixed(N) = % of mutations where test_accuracy >= threshold.

Optionally broken down by mutation order (from aggregated_mutations.json).

Loads GP/BFS final performance from their result directories to show as
horizontal reference lines.

Usage:
    python plot_spec_size.py \
        --saved-data-dir saved_data \
        --gp-results-dir ../gp_baseline/results \
        --exhaustive-results-dir ../gp_baseline/exhaustive_results \
        --output-dir plots
"""

import json
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


N_VALUES = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 40000]


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
        saved_data_dir/{program}/n_{N}/{loss_fn_name}/job_{job_id}/test_results.json

    Returns a dict: N -> list of result dicts.
    """
    results_by_n: Dict[int, List[dict]] = {n: [] for n in n_values}

    for n in n_values:
        pattern = f"*/n_{n}/{loss_fn_name}/*/test_results.json"
        for result_file in sorted(saved_data_dir.glob(pattern)):
            try:
                with open(result_file) as f:
                    rec = json.load(f)
                results_by_n[n].append(rec)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not read {result_file}: {e}")

    return results_by_n


def compute_pct_fixed(results: List[dict], threshold: float = 1.0) -> Optional[float]:
    """Return % of results where test_accuracy >= threshold, or None if empty."""
    if not results:
        return None
    fixed = sum(1 for r in results if r.get("test_accuracy", 0.0) >= threshold)
    return 100.0 * fixed / len(results)


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
# Plotting
# ---------------------------------------------------------------------------

def plot_spec_size_ablation(
    results_by_n: Dict[int, List[dict]],
    mutation_orders: Dict[str, int],
    threshold: float,
    n_values: List[int],
    gp_pct: Optional[float],
    bfs_pct: Optional[float],
    output_dir: Path,
    by_mutation_order: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Overall curve ---------------------------------------------------
    overall_x: List[int] = []
    overall_y: List[float] = []
    for n in n_values:
        pct = compute_pct_fixed(results_by_n[n], threshold)
        if pct is not None:
            overall_x.append(n)
            overall_y.append(pct)

    if not overall_x:
        print("No results found; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        overall_x,
        overall_y,
        marker="o",
        label=f"GBPR (n={sum(len(results_by_n[n]) for n in n_values if results_by_n[n])} total)",
        color="#2ca02c",
        linewidth=2,
        markersize=7,
    )

    # Reference lines for GP and BFS
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
    ax.set_title("Specification Size Ablation", fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_xticks(overall_x)
    ax.set_xticklabels([str(n) for n in overall_x], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "spec_size_ablation.pdf"
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved {out_path}")

    # ----- Per-mutation-order breakdown ------------------------------------
    if by_mutation_order and mutation_orders:
        # Determine which orders exist
        all_orders = sorted(set(mutation_orders.values()))

        # Build results_by_n_by_order
        # results_by_n_by_order[order][n] = list of results
        results_by_order: Dict[int, Dict[int, List[dict]]] = {
            order: {n: [] for n in n_values} for order in all_orders
        }
        for n in n_values:
            for rec in results_by_n[n]:
                job_id = rec.get("job_id")
                order = mutation_orders.get(job_id)
                if order is not None:
                    results_by_order[order][n].append(rec)

        fig, ax = plt.subplots(figsize=(9, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(all_orders)))
        for order, color in zip(all_orders, colors):
            xs: List[int] = []
            ys: List[float] = []
            for n in n_values:
                pct = compute_pct_fixed(results_by_order[order][n], threshold)
                if pct is not None:
                    xs.append(n)
                    ys.append(pct)
            if xs:
                n_jobs = len(results_by_order[order][n_values[-1]])
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    label=f"Order {order} (n={n_jobs})",
                    color=color,
                    linewidth=1.8,
                    markersize=6,
                )

        # Overall GBPR curve as reference
        ax.plot(
            overall_x,
            overall_y,
            marker="s",
            label="GBPR overall",
            color="#2ca02c",
            linewidth=2.5,
            markersize=7,
            linestyle="-.",
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
        ax.set_title("Specification Size Ablation by Mutation Order", fontsize=13)
        ax.set_ylim(0, 100)
        ax.set_xticks(overall_x)
        ax.set_xticklabels([str(n) for n in overall_x], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        out_path_order = output_dir / "spec_size_ablation_by_order.pdf"
        plt.savefig(out_path_order, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"Saved {out_path_order}")


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
    default=1.0,
    show_default=True,
    help="Accuracy threshold to count a mutation as fixed",
)
@click.option(
    "--loss-fn-name",
    default="cross_entropy_loss",
    show_default=True,
    help="Loss function subdirectory name",
)
@click.option(
    "--by-mutation-order/--no-by-mutation-order",
    default=True,
    show_default=True,
    help="Also generate a per-mutation-order breakdown plot",
)
def main(
    saved_data_dir,
    gp_results_dir,
    exhaustive_results_dir,
    output_dir,
    threshold,
    loss_fn_name,
    by_mutation_order,
):
    """Plot GBPR % fixed vs. number of training samples (specification size ablation)."""
    script_dir = Path(__file__).parent
    saved_data_path = Path(saved_data_dir) if Path(saved_data_dir).is_absolute() else script_dir / saved_data_dir
    output_path = Path(output_dir) if Path(output_dir).is_absolute() else script_dir / output_dir

    print(f"Loading results from {saved_data_path} ...")
    results_by_n = load_spec_size_results(saved_data_path, N_VALUES, loss_fn_name)
    for n in N_VALUES:
        print(f"  N={n:>6}: {len(results_by_n[n])} results")

    # Load mutation orders for breakdown plot
    mutation_orders = load_mutation_orders(script_dir)
    print(f"Loaded mutation orders for {len(mutation_orders)} jobs")

    # Load GP and BFS final performance
    gp_pct: Optional[float] = None
    if gp_results_dir:
        gp_path = Path(gp_results_dir) if Path(gp_results_dir).is_absolute() else script_dir / gp_results_dir
        gp_pct = load_gp_final_pct_fixed(gp_path, threshold)
        if gp_pct is not None:
            print(f"GP final % fixed: {gp_pct:.1f}%")
        else:
            print(f"Warning: no GP results loaded from {gp_path}")

    bfs_pct: Optional[float] = None
    if exhaustive_results_dir:
        bfs_path = Path(exhaustive_results_dir) if Path(exhaustive_results_dir).is_absolute() else script_dir / exhaustive_results_dir
        bfs_pct = load_gp_final_pct_fixed(bfs_path, threshold)
        if bfs_pct is not None:
            print(f"BFS final % fixed: {bfs_pct:.1f}%")
        else:
            print(f"Warning: no BFS results loaded from {bfs_path}")

    plot_spec_size_ablation(
        results_by_n=results_by_n,
        mutation_orders=mutation_orders,
        threshold=threshold,
        n_values=N_VALUES,
        gp_pct=gp_pct,
        bfs_pct=bfs_pct,
        output_dir=output_path,
        by_mutation_order=by_mutation_order,
    )


if __name__ == "__main__":
    main()
