"""
plot_spec_size.py  –  Specification Size Ablation Plots

Plot A — % fixed vs N, one curve per mutation order (spec_size_ablation_by_order.pdf):
  X-axis: N (log scale)
  Y-axis: % bugs fixed (test_accuracy >= threshold)
  Five GBPR curves (orders 1-5) + dashed horizontal reference lines for GP and BFS
  per mutation order.

Plot B — Learning curves at different N (spec_size_curves.pdf):
  X-axis: gradient steps; Y-axis: mean val accuracy. One line per N value.

Job set: 1,135 jobs — the 1,466 N=40,000 train_mutations jobs with shuffle_dyck
excluded entirely (shuffle_dyck only has 1,635 training samples, so it cannot
participate at N>=5,000; excluding it at all N keeps the denominator fixed).
GP and BFS baselines are restricted to the same 1,135 jobs.

Usage (from repo root):
    python experiments/spec_size/plot_spec_size.py \
        --gp-results-dir experiments/gp_baseline/results \
        --exhaustive-results-dir experiments/gp_baseline/exhaustive_results
"""

import json
import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set


N_VALUES = [100, 1000, 5000, 10000, 25000, 40000]
TRAIN_MUTATIONS_N = 40000
BATCH_SIZE = 256
VAL_STEP = 10
THRESHOLDS = [0.99, 0.999, 1.0]
ORDERS = [1, 2, 3, 4, 5]

ORDER_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
}


# ---------------------------------------------------------------------------
# Reference set + job maps
# ---------------------------------------------------------------------------

def load_job_maps(script_dir: Path):
    mutation_path = (script_dir / "../mutation/results/aggregated_mutations.json").resolve()
    agg_data = json.loads(mutation_path.read_text())
    keys = list(agg_data.keys())
    n = len(agg_data[keys[0]])
    rows = [{k: agg_data[k][str(i)] for k in keys} for i in range(n)]
    buggy = [r for r in rows if r.get("execution_result", {}).get("status") == "BUGGY_MODEL"]
    job_to_order = {r["job_id"]: r["mutation_order"] for r in buggy}
    job_to_prog = {r["job_id"]: r["program_name"] for r in buggy}
    return job_to_order, job_to_prog


def load_reference_ids(train_mutations_dir: Path, job_to_prog: Dict, loss_fn: str) -> Set[str]:
    """1,135 non-shuffle_dyck job IDs present in the N=40,000 train_mutations results."""
    ref = set()
    for f in train_mutations_dir.glob(f"*/{loss_fn}/job_*/test_results.json"):
        job_id = json.loads(f.read_text()).get("job_id")
        if job_id and job_to_prog.get(job_id) != "shuffle_dyck":
            ref.add(job_id)
    return ref


# ---------------------------------------------------------------------------
# GBPR data loading
# ---------------------------------------------------------------------------

def load_gbpr_by_order(
    spec_size_dir: Path,
    train_mutations_dir: Path,
    job_to_order: Dict,
    reference_ids: Set[str],
    loss_fn: str,
) -> Dict[int, Dict[int, List[float]]]:
    """Returns dict: N -> order -> list[test_accuracy]."""
    by_n_order: Dict[int, Dict[int, List[float]]] = {n: defaultdict(list) for n in N_VALUES}

    for n in N_VALUES[:-1]:
        for f in spec_size_dir.glob(f"*/n_{n}/seed_*/{loss_fn}/*/test_results.json"):
            rec = json.loads(f.read_text())
            job_id = rec.get("job_id")
            if job_id not in reference_ids:
                continue
            order = job_to_order.get(job_id)
            if order is not None:
                by_n_order[n][order].append(rec.get("test_accuracy", 0.0))

    for f in train_mutations_dir.glob(f"*/{loss_fn}/job_*/test_results.json"):
        rec = json.loads(f.read_text())
        job_id = rec.get("job_id")
        if job_id not in reference_ids:
            continue
        order = job_to_order.get(job_id)
        if order is not None:
            by_n_order[TRAIN_MUTATIONS_N][order].append(rec.get("test_accuracy", 0.0))

    return by_n_order


def load_val_accs_by_n(
    spec_size_dir: Path,
    train_mutations_dir: Path,
    reference_ids: Set[str],
    job_to_order: Dict,
    loss_fn: str,
) -> Dict[int, List[np.ndarray]]:
    """Returns dict: N -> list of val_accs arrays (reference set only)."""
    by_n: Dict[int, List[np.ndarray]] = {n: [] for n in N_VALUES}

    for n in N_VALUES[:-1]:
        for f in spec_size_dir.glob(f"*/n_{n}/seed_*/{loss_fn}/*/val_accs.npy"):
            # derive job_id from path: .../job_{job_id}/val_accs.npy
            job_id_str = f.parent.name  # "job_{id}"
            # find matching test_results to get job_id
            tr = f.parent / "test_results.json"
            if not tr.exists():
                continue
            job_id = json.loads(tr.read_text()).get("job_id")
            if job_id not in reference_ids:
                continue
            try:
                by_n[n].append(np.load(f))
            except (OSError, ValueError):
                pass

    for f in train_mutations_dir.glob(f"*/{loss_fn}/job_*/val_accs.npy"):
        tr = f.parent / "test_results.json"
        if not tr.exists():
            continue
        job_id = json.loads(tr.read_text()).get("job_id")
        if job_id not in reference_ids:
            continue
        try:
            by_n[TRAIN_MUTATIONS_N].append(np.load(f))
        except (OSError, ValueError):
            pass

    return by_n


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def load_baseline_by_order(
    results_dir: Path,
    job_to_order: Dict,
    reference_ids: Set[str],
) -> Dict[int, List[float]]:
    """Returns dict: order -> list[best_test_acc], restricted to reference_ids."""
    by_order: Dict[int, List[float]] = defaultdict(list)
    for f in results_dir.glob("*.json"):
        rec = json.loads(f.read_text())
        job_id = rec.get("job_id")
        if job_id not in reference_ids:
            continue
        order = job_to_order.get(job_id)
        hist = rec.get("program_history", [])
        best = max((e.get("best_test_acc", 0.0) or 0.0) for e in hist) if hist else 0.0
        if order is not None:
            by_order[order].append(best)
    return by_order


# ---------------------------------------------------------------------------
# Plot A — two-panel paper figure
# ---------------------------------------------------------------------------

def pct_fixed(values: List[float], thr: float) -> Optional[float]:
    if not values:
        return None
    return 100.0 * sum(1 for v in values if v >= thr) / len(values)


# Sequential colour palette: light (order 1) → dark (order 5)
ORDER_PALETTE = [
    "#a8d8f0",  # order 1 — light blue
    "#6baed6",  # order 2
    "#3182bd",  # order 3
    "#08519c",  # order 4
    "#08306b",  # order 5 — dark blue
]


def _thresh_suffix(threshold: float) -> str:
    if abs(threshold - 0.99) < 1e-6:
        return ""
    return "_thresh" + f"{threshold:.3f}".rstrip("0").rstrip(".")


def plot_paper_figure(
    gbpr_by_n_order: Dict[int, Dict[int, List[float]]],
    gp_by_order: Optional[Dict[int, List[float]]],
    bfs_by_order: Optional[Dict[int, List[float]]],
    threshold: float,
    output_dir: Path,
    orders: List[int] = ORDERS,
):
    """Two separate PDFs (fig:spec-size).

    spec_size_passrate.pdf  — GBPR pass rate vs N, one curve per mutation order.
    spec_size_advantage.pdf — GBPR advantage over max(GP, BFS) in pp; dashed y=0 line.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    x_ticks = N_VALUES
    x_labels = ["100", "1k", "5k", "10k", "25k", "40k"]
    suf = _thresh_suffix(threshold)

    # ── Figure 1: raw pass rate ────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(5.5, 4.2))
    for i, order in enumerate(orders):
        color = ORDER_PALETTE[i]
        ys = [pct_fixed(gbpr_by_n_order[n][order], threshold) or float("nan") for n in N_VALUES]
        ax1.plot(N_VALUES, ys, marker="o", color=color, linewidth=2, markersize=5,
                 label=f"Order {order}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Specification size $N$", fontsize=11)
    ax1.set_ylabel("Pass rate (\\%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    out1 = output_dir / f"spec_size_passrate{suf}.pdf"
    plt.savefig(out1, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved {out1}")

    # ── Figure 2: GBPR − max(GP, BFS) ─────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.2))
    for i, order in enumerate(orders):
        color = ORDER_PALETTE[i]
        gp_p = pct_fixed(gp_by_order.get(order, []), threshold) if gp_by_order else None
        bfs_p = pct_fixed(bfs_by_order.get(order, []), threshold) if bfs_by_order else None
        if gp_p is None and bfs_p is None:
            continue
        best_sym = max(v for v in [gp_p, bfs_p] if v is not None)
        ys = []
        for n in N_VALUES:
            p = pct_fixed(gbpr_by_n_order[n][order], threshold)
            ys.append((p - best_sym) if p is not None else float("nan"))
        ax2.plot(N_VALUES, ys, marker="o", color=color, linewidth=2, markersize=5,
                 label=f"Order {order}")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.3, label="Symbolic baseline")
    ax2.set_xscale("log")
    ax2.set_xlabel("Specification size $N$", fontsize=11)
    ax2.set_ylabel("GBPR $-$ best symbolic (pp)", fontsize=11)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    out2 = output_dir / f"spec_size_advantage{suf}.pdf"
    plt.savefig(out2, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved {out2}")


# ---------------------------------------------------------------------------
# Plot B — Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    val_accs_by_n: Dict[int, List[np.ndarray]],
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    val_step: int = VAL_STEP,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not any(val_accs_by_n[n] for n in N_VALUES):
        print("No val_accs.npy files found; skipping learning curves.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_VALUES)))

    for n, color in zip(N_VALUES, colors):
        arrays = val_accs_by_n[n]
        if not arrays:
            continue
        steps_per_epoch = max(1, n // batch_size)
        min_len = min(len(a) for a in arrays)
        if min_len == 0:
            continue
        mean_acc = np.mean(np.stack([a[:min_len] for a in arrays], axis=0), axis=0)
        steps = np.arange(len(mean_acc)) * val_step * steps_per_epoch
        ax.plot(steps, mean_acc * 100, label=f"N={n} ({len(arrays)} jobs)",
                color=color, linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Gradient steps", fontsize=12)
    ax.set_ylabel("Mean val accuracy (%)", fontsize=12)
    ax.set_title("Learning Curves by Specification Size", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = output_dir / "spec_size_curves.pdf"
    plt.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--saved-data-dir", default="experiments/spec_size/saved_data", show_default=True)
@click.option("--train-mutations-saved-data-dir",
              default="experiments/train_mutations/saved_data", show_default=True)
@click.option("--gp-results-dir", default=None,
              help="Directory with GP baseline per-job JSON results")
@click.option("--exhaustive-results-dir", default=None,
              help="Directory with BFS/exhaustive per-job JSON results")
@click.option("--output-dir", default="experiments/spec_size/plots", show_default=True)
@click.option("--threshold", type=float, default=None,
              help="Single threshold (default: run all of 0.99, 0.999, 1.0)")
@click.option("--loss-fn-name", default="cross_entropy_loss", show_default=True)
def main(
    saved_data_dir,
    train_mutations_saved_data_dir,
    gp_results_dir,
    exhaustive_results_dir,
    output_dir,
    threshold,
    loss_fn_name,
):
    """Plot GBPR % fixed vs. N (spec size ablation), excluding shuffle_dyck throughout."""
    script_dir = Path(__file__).parent

    def resolve(p):
        return Path(p).resolve()

    spec_size_dir = resolve(saved_data_dir)
    train_mutations_dir = resolve(train_mutations_saved_data_dir)
    output_path = resolve(output_dir)

    # Load job maps and reference set
    job_to_order, job_to_prog = load_job_maps(script_dir)
    reference_ids = load_reference_ids(train_mutations_dir, job_to_prog, loss_fn_name)
    print(f"Reference set: {len(reference_ids)} jobs (shuffle_dyck excluded at all N)")

    # Load GBPR results
    gbpr = load_gbpr_by_order(
        spec_size_dir, train_mutations_dir, job_to_order, reference_ids, loss_fn_name
    )
    for n in N_VALUES:
        total = sum(len(gbpr[n][o]) for o in ORDERS)
        print(f"  N={n:>6}: {total} results")

    # Load baselines
    gp_by_order = None
    if gp_results_dir:
        gp_path = resolve(gp_results_dir)
        gp_by_order = load_baseline_by_order(gp_path, job_to_order, reference_ids)
        total_gp = sum(len(v) for v in gp_by_order.values())
        print(f"GP baseline: {total_gp} results (restricted to reference set)")

    bfs_by_order = None
    if exhaustive_results_dir:
        bfs_path = resolve(exhaustive_results_dir)
        bfs_by_order = load_baseline_by_order(bfs_path, job_to_order, reference_ids)
        total_bfs = sum(len(v) for v in bfs_by_order.values())
        print(f"BFS baseline: {total_bfs} results (restricted to reference set)")

    thresholds_to_run = [threshold] if threshold is not None else THRESHOLDS

    for thr in thresholds_to_run:
        plot_paper_figure(gbpr, gp_by_order, bfs_by_order, thr, output_path)

    # Learning curves (threshold-independent)
    val_accs = load_val_accs_by_n(
        spec_size_dir, train_mutations_dir, reference_ids, job_to_order, loss_fn_name
    )
    plot_learning_curves(val_accs, output_path)


if __name__ == "__main__":
    main()
