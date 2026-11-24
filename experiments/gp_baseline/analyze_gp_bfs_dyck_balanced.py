#!/usr/bin/env python3
"""Recompute GP/BFS Dyck-1/Dyck-2 results with class-imbalance-aware metrics.

Mirrors experiments/train_mutations/analyze_dyck_balanced.py, but for the
symbolic GP and BFS baselines: rebuilds each method's best-found RASP program
from its stored source and re-evaluates it on the held-out test set, reporting
positive-example exact-match accuracy and token-level balanced accuracy/macro-F1
alongside the original full-sequence exact-match accuracy already stored in the
result JSON (as test_acc). This is needed to check whether GBPR's claimed
order-4/5 advantage over GP/BFS survives once shuffle_dyck2's inflated pass
rate (99.0% GP, 95.8% BFS, both driven by output-class imbalance, see
experiments/train_mutations/analyze_dyck_balanced.py) is corrected for all
three methods on an apples-to-apples basis.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd

from experiments.gp_baseline.gp_core import (
    build_program_from_source,
    canonicalize_for_dataset,
)
from src.functions import load_dataset, getAcceptedNamesAndInput

PROGRAMS = ["shuffle_dyck", "shuffle_dyck2"]
METHOD_DIRS = {
    "GP": "results_1000",
    "BFS": "exhaustive_results_1000",
}


def evaluate_source_balanced(source_code, program_name, max_length, accepted_inputs, test_data):
    program = build_program_from_source(source_code, program_name, max_length, accepted_inputs)

    all_true = []
    all_pred = []
    seq_exact = []
    seq_has_positive = []

    for input_seq, target_seq in test_data:
        x = input_seq[1:]
        y = target_seq[1:]
        try:
            pred = list(program(x))
        except Exception:
            pred = [None] * len(y)

        y_arr = np.array(y)
        # Treat any non-comparable/failed prediction position as incorrect (not equal to label)
        pred_arr = np.array([p if p in (0, 1) else -1 for p in pred]) if len(pred) == len(y) else np.full_like(y_arr, -1)

        all_true.append(y_arr)
        all_pred.append(pred_arr)
        seq_exact.append(bool(np.array_equal(y_arr, pred_arr)))
        seq_has_positive.append(bool(np.any(y_arr == 1)))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    balanced_acc = np.nanmean([tpr, tnr])

    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_pos = 2 * precision_pos * tpr / (precision_pos + tpr) if (precision_pos + tpr) > 0 else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_neg = 2 * precision_neg * tnr / (precision_neg + tnr) if (precision_neg + tnr) > 0 else 0.0
    macro_f1 = np.nanmean([f1_pos, f1_neg])

    seq_exact = np.array(seq_exact)
    seq_has_positive = np.array(seq_has_positive)
    n_pos_seqs = int(seq_has_positive.sum())
    pos_seq_exact_acc = float(seq_exact[seq_has_positive].mean()) if n_pos_seqs > 0 else float("nan")

    return {
        "full_seq_exact_match_acc": float(seq_exact.mean()),
        "positive_seq_exact_match_acc": pos_seq_exact_acc,
        "token_balanced_accuracy": float(balanced_acc),
        "token_macro_f1": float(macro_f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent.parent.parent / "data"))
    parser.add_argument("--program", choices=PROGRAMS, default=None,
                         help="Restrict to a single program (default: both).")
    parser.add_argument("--out", type=str, default=str(
        Path(__file__).parent / "analysis_outputs" / "gp_bfs_dyck_balanced_metrics.csv"
    ))
    args = parser.parse_args()

    accepted_inputs_map = getAcceptedNamesAndInput()
    data_dir = Path(args.data_dir)

    programs = [args.program] if args.program else PROGRAMS

    rows = []
    for method, results_dir in METHOD_DIRS.items():
        for program_name in programs:
            dataset_name = canonicalize_for_dataset(program_name)
            accepted_inputs = accepted_inputs_map[dataset_name]
            test_data = load_dataset(data_dir, dataset_name, split_name="test")
            max_length = max(len(inp) for inp, _ in test_data)

            files = sorted(glob.glob(str(Path(__file__).parent / results_dir / f"{dataset_name}_*.json")))
            print(f"{method} / {program_name}: {len(files)} jobs", flush=True)
            for i, f in enumerate(files):
                d = json.load(open(f))
                job_id = Path(f).stem.replace(f"{dataset_name}_", "")
                best_source = d.get("best_source")
                stored_test_acc = d.get("test_acc")
                if not best_source:
                    print(f"  [{i}] {job_id} skipped: no best_source", flush=True)
                    continue
                try:
                    metrics = evaluate_source_balanced(
                        best_source, program_name, max_length, accepted_inputs, test_data
                    )
                except Exception as e:
                    print(f"  [{i}] {job_id} FAILED: {e}", flush=True)
                    continue
                row = {
                    "method": method,
                    "program_name": program_name,
                    "job_id": job_id,
                    "stored_test_acc": stored_test_acc,
                    **metrics,
                }
                rows.append(row)
                if (i + 1) % 25 == 0:
                    print(f"  [{method}/{program_name}] {i + 1}/{len(files)} done", flush=True)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    for (method, program_name), g in df.groupby(["method", "program_name"]):
        n = len(g)
        print(f"\n=== {method} / {program_name} (n={n}) ===")
        print("Mean full-seq exact-match acc:      %.4f" % g["full_seq_exact_match_acc"].mean())
        print("Mean positive-seq exact-match acc:  %.4f" % g["positive_seq_exact_match_acc"].mean())
        print("Mean token balanced accuracy:        %.4f" % g["token_balanced_accuracy"].mean())
        fixed_fullseq = (g["full_seq_exact_match_acc"] >= 0.99).mean() * 100
        fixed_posseq = (g["positive_seq_exact_match_acc"] >= 0.99).mean() * 100
        fixed_balanced = (g["token_balanced_accuracy"] >= 0.99).mean() * 100
        print("%% fixed full-seq >=99%%:      %.1f%%" % fixed_fullseq)
        print("%% fixed positive-seq >=99%%:  %.1f%%" % fixed_posseq)
        print("%% fixed balanced-acc >=99%%:  %.1f%%" % fixed_balanced)


if __name__ == "__main__":
    main()
