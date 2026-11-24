"""Freeze the raw results behind the TMLR submission into small tidy tables.

One-shot archival script. It reads the *legacy* run-directory layout (as of tag
`tmlr-submission-1`, where per-run directories with model weights lived inside
`experiments/*/saved_data*`) and writes `results/v1_tmlr/`. After the history
rewrite the raw directories are no longer in the repository; to re-run this
script, extract the raw-run archive (see README) into a checkout of the tag.

Usage:
    python scripts/freeze_v1_results.py [--repo-root PATH] [--skip-weight-stats]
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOSS_FN = "cross_entropy_loss"


def _curves(job_dir: Path, job_key: dict) -> tuple[list[dict], list[dict]]:
    """Per-epoch training curves; val_acc is only recorded every 10 epochs."""
    arrs = {}
    for name in ("train_losses", "train_accs", "val_losses", "val_accs"):
        p = job_dir / f"{name}.npy"
        arrs[name] = np.load(p, allow_pickle=True).reshape(-1) if p.exists() else np.array([])
    n = len(arrs["train_losses"])
    rows = []
    for e in range(n):
        rows.append(
            {
                **job_key,
                "epoch": e,
                "train_loss": float(arrs["train_losses"][e]),
                "train_acc": float(arrs["train_accs"][e]) if e < len(arrs["train_accs"]) else np.nan,
                "val_loss": float(arrs["val_losses"][e]) if e < len(arrs["val_losses"]) else np.nan,
            }
        )
    # val_accs[i] corresponds to validation evaluation i (every VAL_STEP=10 epochs)
    val_rows = [
        {**job_key, "val_eval": i, "val_acc": float(v)} for i, v in enumerate(arrs["val_accs"])
    ]
    return rows, val_rows


def _run_summary(job_dir: Path) -> dict:
    tr = json.loads((job_dir / "test_results.json").read_text())
    tl = job_dir / "train_losses.npy"
    va = job_dir / "val_accs.npy"
    n_epochs = len(np.load(tl, allow_pickle=True)) if tl.exists() else None
    vals = np.load(va, allow_pickle=True).reshape(-1) if va.exists() else np.array([])
    return {
        "job_id": tr["job_id"],
        "program": tr["program_name"],
        "test_accuracy": tr["test_accuracy"],
        "test_samples": tr.get("test_samples"),
        "n_epochs": n_epochs,
        "n_val_evals": len(vals),
        "best_val_acc": float(vals.max()) if len(vals) else np.nan,
        "final_val_acc": float(vals[-1]) if len(vals) else np.nan,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"  wrote {path.relative_to(path.parents[2])} ({len(df)} rows)")


# --------------------------------------------------------------------------- mutants


def freeze_mutants(root: Path, out: Path) -> pd.DataFrame:
    mut_dir = root / "experiments/mutation"
    if not mut_dir.exists():  # layout after the cleanup
        mut_dir = root / "raspbugs"
    agg = pd.read_json(mut_dir / "results/aggregated_mutations.json")
    raspbugs = pd.read_json(mut_dir / "raspbugs.json")
    in_raspbugs = set(raspbugs["job_id"])
    df = pd.DataFrame(
        {
            "job_id": agg["job_id"],
            "program": agg["program_name"],
            "mutation_order": agg["mutation_order"],
            "operators": agg["operators"].apply(
                lambda ops: ";".join(o.replace("operators/", "") for o in ops)
            ),
            "status": agg["execution_result"].apply(lambda e: e.get("status")),
            "oracle_accuracy_50": agg["execution_result"].apply(lambda e: e.get("accuracy")),
        }
    )
    df["in_raspbugs_v1"] = df["job_id"].isin(in_raspbugs)
    _write_csv(df, out / "mutants.csv")

    # The benchmark manifest: every buggy mutant with its source, so mutants can be
    # recompiled without the 56 MB cosmic-ray aggregate. Written next to the pipeline.
    buggy = agg[df["status"] == "BUGGY_MODEL"]
    src_path = root / "raspbugs" / "raspbugs_v1.jsonl.gz"
    src_path.parent.mkdir(exist_ok=True)
    with gzip.open(src_path, "wt") as f:
        for _, r in buggy.iterrows():
            f.write(
                json.dumps(
                    {
                        "job_id": r["job_id"],
                        "program": r["program_name"],
                        "mutation_order": int(r["mutation_order"]),
                        "operators": [o.replace("operators/", "") for o in r["operators"]],
                        "in_raspbugs_v1": r["job_id"] in in_raspbugs,
                        "oracle_accuracy_50": r["execution_result"].get("accuracy"),
                        "diff": r["diff"],
                        "source": r["program_source_after"],
                        "mutations": r["mutation_object"].get("mutations"),
                    }
                )
                + "\n"
            )
    print(f"  wrote {src_path} ({len(buggy)} buggy mutants)")
    return df


# --------------------------------------------------------------------------- GBPR runs


def freeze_gbpr_tree(job_dirs: list[tuple[dict, Path]], out_dir: Path, with_curves: bool) -> None:
    runs, curves, val_curves = [], [], []
    for extra, job_dir in job_dirs:
        if not (job_dir / "test_results.json").exists():
            continue
        s = _run_summary(job_dir)
        row = {**s, **extra}
        runs.append(row)
        if with_curves:
            key = {"job_id": s["job_id"], **{k: v for k, v in extra.items()}}
            c, v = _curves(job_dir, key)
            curves.extend(c)
            val_curves.extend(v)
    _write_csv(pd.DataFrame(runs), out_dir / "runs.csv")
    if with_curves:
        _write_csv(pd.DataFrame(curves), out_dir / "curves.csv.gz")
        _write_csv(pd.DataFrame(val_curves), out_dir / "val_curves.csv.gz")


def main_repair_dirs(root: Path):
    base = root / "experiments/train_mutations/saved_data"
    return [({}, d) for d in sorted(base.glob(f"*/{LOSS_FN}/job_*"))]


def spec_size_dirs(root: Path):
    base = root / "experiments/spec_size/saved_data"
    out = []
    for d in sorted(base.glob(f"*/n_*/seed_*/{LOSS_FN}/job_*")):
        n = int(d.parents[2].name.removeprefix("n_"))
        seed = int(d.parents[1].name.removeprefix("seed_"))
        out.append(({"n_train": n, "seed": seed}, d))
    return out


def paired_init_dirs(root: Path):
    base = root / "experiments/train_mutations/saved_data_paired_init"
    out = []
    for d in sorted(base.glob(f"*/job_*/seed_*/{LOSS_FN}")):
        seed = int(d.parent.name.removeprefix("seed_"))
        out.append(({"init": "xavier", "seed": seed}, d))
    return out


# --------------------------------------------------------------------------- baselines


def freeze_baseline(results_dir: Path, method: str, budget: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, hist = [], []
    for fp in sorted(results_dir.glob("*.json")):
        rec = json.loads(fp.read_text())
        if "job_id" not in rec:
            continue
        h = rec.get("program_history", [])
        runs.append(
            {
                "method": method,
                "budget": budget,
                "job_id": rec["job_id"],
                "program_file": rec["program_name"],
                "status": rec.get("status"),
                "num_programs": rec.get("num_programs"),
                "num_duplicates": rec.get("num_duplicates"),
                "num_compile_failures": rec.get("num_compile_failures"),
                "train_acc": rec.get("train_acc"),
                "test_acc": rec.get("test_acc"),
                "max_best_test_acc": max(
                    ((e.get("best_test_acc") or 0.0) for e in h), default=0.0
                ),
                "duration_s": rec.get("duration_s"),
                "seed": rec.get("seed"),
                "history_len": len(h),
                "max_step": max((int(e.get("step", 0)) for e in h), default=0),
            }
        )
        # Keep only change points of the best-so-far values (+ first/last entry).
        prev = None
        for i, e in enumerate(h):
            key = (e.get("best_train_acc"), e.get("best_test_acc"))
            if key != prev or i == len(h) - 1:
                hist.append(
                    {
                        "method": method,
                        "budget": budget,
                        "job_id": rec["job_id"],
                        "step": e.get("step"),
                        "generation": e.get("generation", e.get("depth")),
                        "status": e.get("status"),
                        "train_acc": e.get("train_acc"),
                        "test_acc": e.get("test_acc"),
                        "best_train_acc": e.get("best_train_acc"),
                        "best_test_acc": e.get("best_test_acc"),
                    }
                )
                prev = key
    return pd.DataFrame(runs), pd.DataFrame(hist)


# --------------------------------------------------------------------------- weight stats


def _vector_stats(params: dict) -> tuple[float, float, float, int]:
    flat = np.concatenate(
        [np.asarray(t).ravel() for layer in params.values() for t in layer.values()]
    )
    return float((flat == 0).mean()), float(flat.std()), float(np.abs(flat).max()), int(flat.size)


def freeze_weight_stats(root: Path, out_dir: Path) -> None:
    sys.path.insert(0, str(root))
    from experiments.mutation.load_mutations import load_buggy_models  # legacy layout

    sel = json.loads(
        (root / "experiments/train_mutations/saved_data_paired_init/selected_jobs.json").read_text()
    )
    rows = []
    for program, job_ids in sel["selection"].items():
        for job_id in job_ids:
            model = load_buggy_models(max_length=10, program_name=program, job_id=job_id)[job_id]
            z, s, m, n = _vector_stats(model.model.params)
            rows.append(dict(job_id=job_id, program=program, state="compiled", seed=None,
                             zero_frac=z, std=s, max_abs=m, n_params=n))
            p = root / f"experiments/train_mutations/saved_data/{program}/{LOSS_FN}/job_{job_id}/model.npy"
            if p.exists():
                z, s, m, n = _vector_stats(np.load(p, allow_pickle=True).item())
                rows.append(dict(job_id=job_id, program=program, state="repaired_from_compiled",
                                 seed=None, zero_frac=z, std=s, max_abs=m, n_params=n))
            for sd in sorted((root / "experiments/train_mutations/saved_data_paired_init"
                              / program / f"job_{job_id}").glob("seed_*")):
                p = sd / LOSS_FN / "model.npy"
                if p.exists():
                    z, s, m, n = _vector_stats(np.load(p, allow_pickle=True).item())
                    rows.append(dict(job_id=job_id, program=program, state="trained_from_xavier",
                                     seed=int(sd.name.removeprefix("seed_")),
                                     zero_frac=z, std=s, max_abs=m, n_params=n))
            print(f"    weight stats {program} {job_id}", flush=True)
    _write_csv(pd.DataFrame(rows), out_dir / "weight_stats.csv")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-weight-stats", action="store_true")
    args = ap.parse_args()
    root = args.repo_root.resolve()
    out = (args.out or root / "results/v1_tmlr").resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("mutants")
    freeze_mutants(root, out)

    print("main_repair")
    freeze_gbpr_tree(main_repair_dirs(root), out / "main_repair", with_curves=True)

    print("symbolic_baselines")
    gb = root / "experiments/gp_baseline"
    for method, d, budget, sub in [
        ("gp", gb / "results_1000", 1000, "symbolic_baselines"),
        ("bfs", gb / "exhaustive_results_1000", 1000, "symbolic_baselines"),
        ("gp", gb / "results", 500, "spec_size"),
        ("bfs", gb / "exhaustive_results", 500, "spec_size"),
    ]:
        runs, hist = freeze_baseline(d, method, budget)
        _write_csv(runs, out / sub / f"{method}_budget{budget}_runs.csv")
        _write_csv(hist, out / sub / f"{method}_budget{budget}_history.csv.gz")
    shutil.copy(root / "gbpr_timing.json", out / "symbolic_baselines/gbpr_timing.json")

    print("spec_size")
    freeze_gbpr_tree(spec_size_dirs(root), out / "spec_size", with_curves=False)

    print("init_control")
    freeze_gbpr_tree(paired_init_dirs(root), out / "init_control", with_curves=True)
    shutil.copy(
        root / "experiments/train_mutations/saved_data_paired_init/selected_jobs.json",
        out / "init_control/selected_jobs.json",
    )
    if not args.skip_weight_stats:
        freeze_weight_stats(root, out / "init_control")

    print("dyck")
    (out / "dyck").mkdir(exist_ok=True)
    shutil.copy(
        root / "experiments/train_mutations/analysis_outputs/dyck_balanced_metrics.csv",
        out / "dyck/gbpr_metrics.csv",
    )
    shutil.copy(
        root / "experiments/gp_baseline/analysis_outputs/gp_bfs_dyck_balanced_metrics_merged.csv",
        out / "dyck/baseline_metrics.csv",
    )


if __name__ == "__main__":
    main()
