"""Accuracy and repair metrics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rasping.config import THRESHOLD


def exact_match_accuracy(logits, x, y, pad_token):
    """Fraction of sequences whose argmax prediction matches `y` on every counted position.

    The BOS position and padding positions (where `x == pad_token`) are ignored.
    """
    pred = jnp.argmax(logits, axis=-1)
    mask = jnp.ones_like(x).at[:, 0].set(0)
    mask = jnp.where(x != pad_token, mask, 0)
    return jnp.mean(jnp.all(pred * mask == y * mask, axis=[-1]).astype(float))


def is_repaired(accuracy, threshold: float = THRESHOLD):
    """The repair criterion: held-out exact-match accuracy >= threshold (elementwise)."""
    return np.asarray(accuracy) >= threshold


def pass_rate(accuracies, threshold: float = THRESHOLD) -> float:
    """Percentage of programs meeting the repair criterion."""
    acc = np.asarray(accuracies, dtype=float)
    return float(100.0 * np.mean(acc >= threshold)) if acc.size else float("nan")


def dyck_metrics(y_true: list, y_pred: list) -> dict:
    """Imbalance-aware metrics for the binary Dyck programs (paper App. C).

    `y_true` / `y_pred` are per-sequence arrays of token labels over the counted positions
    (BOS and padding removed); a failed prediction should use a label outside {0, 1}.
    A sequence is "positive" (balanced string) if any of its labels is 1.
    """
    t, p = np.concatenate(y_true), np.concatenate(y_pred)
    tp, tn = int(np.sum((t == 1) & (p == 1))), int(np.sum((t == 0) & (p == 0)))
    # v1 definition: a failed prediction (label outside {0,1}) on a positive token is
    # neither a TP nor a FN, which inflates token recall for crashing programs (F28).
    fp, fn = int(np.sum((t == 0) & (p == 1))), int(np.sum((t == 1) & (p == 0)))
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    prec_pos = tp / (tp + fp) if tp + fp else 0.0
    prec_neg = tn / (tn + fn) if tn + fn else 0.0
    f1_pos = 2 * prec_pos * tpr / (prec_pos + tpr) if prec_pos + tpr else 0.0
    f1_neg = 2 * prec_neg * tnr / (prec_neg + tnr) if prec_neg + tnr else 0.0
    exact = np.array([np.array_equal(a, b) for a, b in zip(y_true, y_pred)])
    positive = np.array([np.any(a == 1) for a in y_true])
    return {
        "n_test_sequences": len(y_true),
        "n_positive_sequences": int(positive.sum()),
        "full_seq_exact_match_acc": float(exact.mean()),
        "positive_seq_exact_match_acc": float(exact[positive].mean()) if positive.any() else float("nan"),
        "token_balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "token_macro_f1": float(np.nanmean([f1_pos, f1_neg])),
        "token_tpr_recall_pos": float(tpr),
        "token_tnr_recall_neg": float(tnr),
    }
