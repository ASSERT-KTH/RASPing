"""
print_review_stats.py  –  Outputs reviewer-facing statistics for the spec_size ablation.

Run from the repo root:
    python experiments/spec_size/print_review_stats.py

shuffle_dyck is excluded at all N values because it has only 1,635 distinct training
samples and cannot be trained at N>=5,000. Including it only at low N would change
the job-set composition across N values, confounding the comparison. Excluding it
everywhere gives a fixed denominator of 1,135 jobs across all N.

Reference set: 1,466 jobs from N=40,000 train_mutations, minus 331 shuffle_dyck jobs
= 1,135 jobs. GP and BFS baselines are restricted to the same 1,135 jobs.
"""

import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

AGG_PATH = REPO_ROOT / "experiments/mutation/results/aggregated_mutations.json"
SPEC_SIZE_DIR = SCRIPT_DIR / "saved_data"
TRAIN_MUTATIONS_DIR = REPO_ROOT / "experiments/train_mutations/saved_data"
GP_DIR = REPO_ROOT / "experiments/gp_baseline/results"
BFS_DIR = REPO_ROOT / "experiments/gp_baseline/exhaustive_results"

LOSS_FN = "cross_entropy_loss"
THRESHOLD = 0.99
N_VALUES = [100, 1000, 5000, 10000, 25000, 40000]


def load_job_maps():
    agg_data = json.loads(AGG_PATH.read_text())
    keys = list(agg_data.keys())
    n = len(agg_data[keys[0]])
    rows = [{k: agg_data[k][str(i)] for k in keys} for i in range(n)]
    buggy = [r for r in rows if r.get("execution_result", {}).get("status") == "BUGGY_MODEL"]
    job_to_order = {r["job_id"]: r["mutation_order"] for r in buggy}
    job_to_prog = {r["job_id"]: r["program_name"] for r in buggy}
    return job_to_order, job_to_prog


EXCLUDED_PROGRAMS = {"shuffle_dyck", "shuffle_dyck2"}


def load_reference_job_ids(job_to_prog):
    """Return the 847 job_ids present in train_mutations results, excluding both Dyck programs."""
    ref_ids = set()
    for f in TRAIN_MUTATIONS_DIR.glob(f"*/{LOSS_FN}/job_*/test_results.json"):
        job_id = json.loads(f.read_text()).get("job_id")
        if job_id is not None and job_to_prog.get(job_id) not in EXCLUDED_PROGRAMS:
            ref_ids.add(job_id)
    return ref_ids


def load_gbpr(job_to_order, reference_ids):
    """Returns dict: N -> mutation_order -> list[float]."""
    by_n_order = {n: defaultdict(list) for n in N_VALUES}

    for n in N_VALUES[:-1]:  # 100..25000
        for f in SPEC_SIZE_DIR.glob(f"*/n_{n}/seed_*/{LOSS_FN}/*/test_results.json"):
            rec = json.loads(f.read_text())
            job_id = rec.get("job_id")
            if job_id not in reference_ids:
                continue
            order = job_to_order.get(job_id)
            if order is not None:
                by_n_order[n][order].append(rec.get("test_accuracy", 0.0))

    for f in TRAIN_MUTATIONS_DIR.glob(f"*/{LOSS_FN}/job_*/test_results.json"):
        rec = json.loads(f.read_text())
        job_id = rec.get("job_id")
        if job_id not in reference_ids:
            continue
        order = job_to_order.get(job_id)
        if order is not None:
            by_n_order[40000][order].append(rec.get("test_accuracy", 0.0))

    return by_n_order


def load_baseline(results_dir, job_to_order, reference_ids):
    """Returns dict: mutation_order -> list[float], restricted to reference_ids."""
    by_order = defaultdict(list)
    for f in Path(results_dir).glob("*.json"):
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


def pct_fixed(values, thr=THRESHOLD):
    if not values:
        return None
    return 100.0 * sum(1 for v in values if v >= thr) / len(values)


def main():
    job_to_order, job_to_prog = load_job_maps()
    reference_ids = load_reference_job_ids(job_to_prog)
    print(f"Reference set: {len(reference_ids)} jobs (shuffle_dyck excluded at all N)")

    gbpr = load_gbpr(job_to_order, reference_ids)
    gp = load_baseline(GP_DIR, job_to_order, reference_ids)
    bfs = load_baseline(BFS_DIR, job_to_order, reference_ids)

    orders = sorted({o for n in N_VALUES for o in gbpr[n]} | set(gp) | set(bfs))

    # ── GBPR table ───────────────────────────────────────────────────────────
    print(f"\n{'':=<70}")
    print(f"  GBPR: % bugs fixed (test_accuracy >= {THRESHOLD}) by N and mutation order")
    print(f"{'':=<70}")
    hdr = f"{'Order':>7}" + "".join(f"  N={n:>6}" for n in N_VALUES)
    print(hdr)
    print("-" * len(hdr))
    for o in orders:
        row = f"{o:>7}"
        for n in N_VALUES:
            p = pct_fixed(gbpr[n][o])
            row += f"  {p:>7.1f}%" if p is not None else f"  {'—':>8}"
        print(row)
    row = f"{'ALL':>7}"
    for n in N_VALUES:
        p = pct_fixed([v for o in orders for v in gbpr[n][o]])
        row += f"  {p:>7.1f}%" if p is not None else f"  {'—':>8}"
    print("-" * len(hdr))
    print(row)

    # ── Baselines ────────────────────────────────────────────────────────────
    print(f"\n{'':=<50}")
    print(f"  Baselines: % bugs fixed (>= {THRESHOLD}) by mutation order")
    print(f"{'':=<50}")
    hdr2 = f"{'Order':>7}  {'GP':>8}  {'BFS':>8}  {'n':>6}"
    print(hdr2)
    print("-" * len(hdr2))
    for o in orders:
        gp_p = pct_fixed(gp[o])
        bfs_p = pct_fixed(bfs[o])
        gp_s = f"{gp_p:>7.1f}%" if gp_p is not None else f"{'—':>8}"
        bfs_s = f"{bfs_p:>7.1f}%" if bfs_p is not None else f"{'—':>8}"
        print(f"{o:>7}  {gp_s}  {bfs_s}  {len(gp[o]):>6}")
    gp_all = pct_fixed([v for o in orders for v in gp[o]])
    bfs_all = pct_fixed([v for o in orders for v in bfs[o]])
    print("-" * len(hdr2))
    print(f"{'ALL':>7}  {gp_all:>7.1f}%  {bfs_all:>7.1f}%")

    # ── Crossover analysis ────────────────────────────────────────────────────
    print(f"\n{'':=<70}")
    print("  Crossover: smallest N at which GBPR first beats GP / BFS")
    print(f"  (threshold={THRESHOLD})")
    print(f"{'':=<70}")
    for o in orders:
        gp_p = pct_fixed(gp[o])
        bfs_p = pct_fixed(bfs[o])
        cross_gp, cross_bfs = "never", "never"
        for n in N_VALUES:
            p = pct_fixed(gbpr[n][o])
            if p is not None:
                if gp_p is not None and cross_gp == "never" and p >= gp_p:
                    cross_gp = f"N={n} (GBPR {p:.1f}% vs GP {gp_p:.1f}%)"
                if bfs_p is not None and cross_bfs == "never" and p >= bfs_p:
                    cross_bfs = f"N={n} (GBPR {p:.1f}% vs BFS {bfs_p:.1f}%)"
        print(f"  Order {o}: vs GP → {cross_gp}  |  vs BFS → {cross_bfs}")

    # ── Sample counts ─────────────────────────────────────────────────────────
    print(f"\n  Sample counts (shuffle_dyck excluded at all N):")
    for n in N_VALUES:
        total = sum(len(gbpr[n][o]) for o in orders)
        print(f"    N={n:>6}: {total} jobs")


if __name__ == "__main__":
    main()
