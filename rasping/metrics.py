"""Accuracy and repair metrics."""

from __future__ import annotations

import numpy as np

from rasping.config import THRESHOLD


def exact_match_accuracy(logits, x, y, pad_token):
    """Fraction of sequences whose argmax prediction matches `y` on every counted position.

    The BOS position and padding positions (where `x == pad_token`) are ignored.
    """
    import jax.numpy as jnp  # lazy: keep this module importable without JAX

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
