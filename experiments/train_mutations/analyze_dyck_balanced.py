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
import signal
import argparse
from pathlib import Path
import csv

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


def ensure_cuda_ready(max_attempts=10, initial_delay=5.0, per_attempt_timeout=60):
    """Force JAX's CUDA backend init here, with retries, in a FRESH subprocess each time.

    On this cluster's shared MIG slices, many processes racing to create a
    CUDA context at once can transiently fail with CUDA_ERROR_DEVICE_UNAVAILABLE.
    Worse: once that happens inside a process, the JAX/XLA client is left in a
    broken state and every subsequent call in that SAME process hangs
    indefinitely instead of raising or succeeding (observed directly: retrying
    the same failed process's second and third JAX calls hung past a 120s
    alarm). So retrying in-process is not safe. Instead, probe readiness in a
    disposable subprocess each attempt; only import jax/return in the parent
    once a probe subprocess reports success.
    """
    import time
    import random
    import subprocess as sp

    probe_code = "import jax.numpy as jnp; jnp.array([1.0]).block_until_ready(); print('OK')"
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            result = sp.run(
                [sys.executable, "-c", probe_code],
                capture_output=True, text=True, timeout=per_attempt_timeout,
            )
            if result.returncode == 0 and "OK" in result.stdout:
                if attempt > 1:
                    print(f"CUDA backend ready after {attempt} attempts", flush=True)
                return
            print(f"CUDA init attempt {attempt}/{max_attempts} failed: {result.stderr[-500:]}", flush=True)
        except sp.TimeoutExpired:
            print(f"CUDA init attempt {attempt}/{max_attempts} timed out after {per_attempt_timeout}s", flush=True)
        if attempt == max_attempts:
            raise RuntimeError(f"CUDA backend never became available after {max_attempts} attempts")
        time.sleep(delay + random.uniform(0, delay))
        delay = min(delay * 1.5, 60.0)

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

    X_test_np = np.array(X_test)
    Y_test_np = np.array(Y_test)
    n = X_test_np.shape[0]
    batch_size = 500
    for start in range(0, n, batch_size):
        x_batch = X_test_np[start:start + batch_size]
        y_batch = Y_test_np[start:start + batch_size]

        logits = forward_fun.apply(model.model.params, jnp.array(x_batch)).unembedded_output
        pred_batch = np.array(jnp.argmax(logits, axis=-1))

        mask = np.ones_like(x_batch)
        mask[:, 0] = 0  # exclude BOS
        valid_batch = np.where(x_batch != pad_token, mask, 0).astype(bool)

        for i in range(x_batch.shape[0]):
            valid = valid_batch[i]
            y_valid = y_batch[i][valid]
            pred_valid = pred_batch[i][valid]

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


class JobTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise JobTimeout("per-job evaluation exceeded time limit")


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
    parser.add_argument("--job-timeout-sec", type=int, default=300,
                         help="Hard wall-clock limit per model (guards against JAX/CUDA hangs after a "
                              "transient device-init failure). Job is skipped, not retried, on timeout.")
    args = parser.parse_args()

    ensure_cuda_ready()

    saved_data_root = Path(__file__).parent / "saved_data"
    loss_fn = "cross_entropy_loss"

    programs = [args.program] if args.program else list(PROGRAM_CONFIGS)

    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / "analysis_outputs" / f"dyck_balanced_metrics_shard{args.shard_index}-of-{args.num_shards}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "program_name", "job_id", "n_test_sequences", "n_positive_sequences",
        "full_seq_exact_match_acc", "positive_seq_exact_match_acc",
        "token_balanced_accuracy", "token_macro_f1",
        "token_tpr_recall_pos", "token_tnr_recall_neg",
    ]

    already_done = set()
    rows = []
    if out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        already_done = set(existing["job_id"])
        if already_done:
            print(f"Resuming: {len(already_done)} jobs already completed in {out_path}", flush=True)

    out_file = open(out_path, "a" if already_done else "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if not already_done:
        writer.writeheader()
    out_file.flush()

    signal.signal(signal.SIGALRM, _raise_timeout)

    for program_name in programs:
        all_job_dirs = sorted((saved_data_root / program_name / loss_fn).iterdir())
        job_dirs = all_job_dirs[args.shard_index::args.num_shards]
        print(f"{program_name}: {len(job_dirs)}/{len(all_job_dirs)} jobs assigned to shard {args.shard_index}/{args.num_shards}", flush=True)
        for i, job_dir in enumerate(job_dirs):
            job_id = job_dir.name.replace("job_", "")
            if job_id in already_done:
                continue
            signal.alarm(args.job_timeout_sec)
            try:
                result = evaluate_job(program_name, job_id, job_dir)
            except JobTimeout as e:
                # This one model hung past the per-job alarm. Could be a
                # genuinely pathological model (slow/non-terminating XLA
                # compile) rather than a device-level problem, so just skip
                # it and move on — do NOT restart the process, or a single
                # bad model would make every retry die on the same job.
                print(f"  [{i}] {job_id} FAILED: {e}", flush=True)
                continue
            except Exception as e:
                print(f"  [{i}] {job_id} FAILED: {e}", flush=True)
                if "unable to initialize backend" in str(e).lower():
                    # A clean, fast CUDA backend-init failure (distinct from a
                    # JobTimeout) has been observed to leave the JAX/XLA client
                    # in a broken state where every later call hangs instead of
                    # raising. Recovering needs a fresh process, not a fresh
                    # job, so exit now and let the launcher retry the whole
                    # shard as a new subprocess (already-completed jobs are
                    # skipped on resume via `already_done`).
                    out_file.close()
                    print("CUDA backend-init failure detected — exiting so the launcher retries this shard "
                          "in a fresh process (in-process recovery is not reliable).", flush=True)
                    sys.exit(2)
                continue
            finally:
                signal.alarm(0)
            if result is None:
                continue
            rows.append(result)
            writer.writerow(result)
            out_file.flush()
            print(f"  [{program_name}] {i + 1}/{len(job_dirs)} done (job {job_id}: full_seq={result['full_seq_exact_match_acc']:.4f})", flush=True)

    out_file.close()
    df = pd.DataFrame(rows)
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
