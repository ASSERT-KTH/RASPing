#!/usr/bin/env python3
"""Recompute Dyck-1/Dyck-2 repair results with class-imbalance-aware metrics.

For each trained (repaired) dyck job, reload the model weights, run inference
on the held-out test set, and report per-token balanced accuracy, macro-F1,
exact-match accuracy restricted to positive (unbalanced-containing) sequences,
alongside the existing full-sequence exact-match accuracy, to see whether the
headline ">=99% fixed" numbers survive once trivial all-zero predictions are
accounted for.
"""
import sys
import os
import json
import argparse
from pathlib import Path

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from src.functions import load_dataset, encodeAndPadData
from experiments.mutation.load_mutations import load_buggy_models

PROGRAM_CONFIGS = {
    "shuffle_dyck": "shuffle_dyck1",
    "shuffle_dyck2": "shuffle_dyck2",
}

MAX_LEN = 10


def evaluate_job(program_name, job_id, output_dir):
    model_path = output_dir / "model.npy"
    if not model_path.exists():
        return None

    model = load_buggy_models(max_length=MAX_LEN, program_name=program_name, job_id=job_id)[job_id]
    trained_params = np.load(str(model_path), allow_pickle=True).item()
    model.model.params = trained_params
    model.setForwardFun()

    data_key = PROGRAM_CONFIGS[program_name]
    data_path = f"{Path(__file__).parent.resolve()}/../../data/"
    test_dataset = load_dataset(data_path, data_key, split_name="test")
    X_test, Y_test = encodeAndPadData(test_dataset, model.raspFunction, model.inputs, MAX_LEN)

    pad_token = model.model.input_encoder.encoding_map["compiler_pad"]

    from src.model import forward_fun

    all_true = []
    all_pred = []
    seq_exact = []
    seq_has_positive = []

    for x, y in zip(X_test, Y_test):
        logits = forward_fun.apply(model.model.params, jnp.array([x])).unembedded_output
        pred = jnp.argmax(logits, axis=-1)[0]

        mask = jnp.ones_like(x)
        mask = mask.at[0].set(0)  # exclude BOS
        valid = jnp.where(x != pad_token, mask, 0).astype(bool)

        y_valid = np.array(y)[np.array(valid)]
        pred_valid = np.array(pred)[np.array(valid)]

        all_true.append(y_valid)
        all_pred.append(pred_valid)
        seq_exact.append(bool(np.all(y_valid == pred_valid)))
        seq_has_positive.append(bool(np.any(y_valid == 1)))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall on positive class
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # recall on negative class
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
        "program_name": program_name,
        "job_id": job_id,
        "n_test_sequences": len(X_test),
        "n_positive_sequences": n_pos_seqs,
        "full_seq_exact_match_acc": float(seq_exact.mean()),
        "positive_seq_exact_match_acc": pos_seq_exact_acc,
        "token_balanced_accuracy": float(balanced_acc),
        "token_macro_f1": float(macro_f1),
        "token_tpr_recall_pos": float(tpr),
        "token_tnr_recall_neg": float(tnr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", choices=list(PROGRAM_CONFIGS), default=None,
                         help="Restrict to a single program (for cluster sharding).")
    parser.add_argument("--shard-index", type=int, default=0,
                         help="This shard's index (0-based), for splitting jobs across parallel workers.")
    parser.add_argument("--num-shards", type=int, default=1,
                         help="Total number of shards. Jobs are split job_dirs[shard_index::num_shards].")
    parser.add_argument("--out", type=str, default=None,
                         help="Output CSV path. Defaults to analysis_outputs/dyck_balanced_metrics_shardX-of-Y.csv")
    args = parser.parse_args()

    saved_data_root = Path(__file__).parent / "saved_data"
    loss_fn = "cross_entropy_loss"
    rows = []

    programs = [args.program] if args.program else list(PROGRAM_CONFIGS)

    for program_name in programs:
        all_job_dirs = sorted((saved_data_root / program_name / loss_fn).iterdir())
        job_dirs = all_job_dirs[args.shard_index::args.num_shards]
        print(f"{program_name}: {len(job_dirs)}/{len(all_job_dirs)} jobs assigned to shard {args.shard_index}/{args.num_shards}", flush=True)
        for i, job_dir in enumerate(job_dirs):
            job_id = job_dir.name.replace("job_", "")
            try:
                result = evaluate_job(program_name, job_id, job_dir)
            except Exception as e:
                print(f"  [{i}] {job_id} FAILED: {e}", flush=True)
                continue
            if result is None:
                continue
            rows.append(result)
            print(f"  [{program_name}] {i + 1}/{len(job_dirs)} done (job {job_id}: full_seq={result['full_seq_exact_match_acc']:.4f})", flush=True)

    df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / "analysis_outputs" / f"dyck_balanced_metrics_shard{args.shard_index}-of-{args.num_shards}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}", flush=True)

    for program_name, g in df.groupby("program_name"):
        print(f"\n=== {program_name} (n={len(g)}) ===")
        print("Mean full-seq exact-match acc:      %.4f" % g["full_seq_exact_match_acc"].mean())
        print("Mean positive-seq exact-match acc:  %.4f" % g["positive_seq_exact_match_acc"].mean())
        print("Mean token balanced accuracy:       %.4f" % g["token_balanced_accuracy"].mean())
        print("Mean token macro-F1:                %.4f" % g["token_macro_f1"].mean())
        fixed_99_fullseq = (g["full_seq_exact_match_acc"] >= 0.99).mean() * 100
        fixed_99_balanced = (g["token_balanced_accuracy"] >= 0.99).mean() * 100
        fixed_99_posseq = (g["positive_seq_exact_match_acc"] >= 0.99).mean() * 100
        print("%% fixed by full-seq exact-match >=99%%:      %.1f%%" % fixed_99_fullseq)
        print("%% fixed by positive-seq exact-match >=99%%:  %.1f%%" % fixed_99_posseq)
        print("%% fixed by token balanced accuracy >=99%%:   %.1f%%" % fixed_99_balanced)


if __name__ == "__main__":
    main()
