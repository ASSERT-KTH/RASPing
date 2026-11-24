"""Correctness losses (Haiku-transformed so they can call the compiled forward pass)."""

import haiku as hk
import jax
import jax.numpy as jnp


@hk.without_apply_rng
@hk.transform
def cross_entropy_loss(x, y, padToken, forward):
    """Mean over sequences of the token-level cross-entropy, BOS and padding masked."""
    logits = forward(x).unembedded_output
    mask = jnp.ones_like(y).at[:, 0].set(0)
    pad_mask = jnp.where(x != padToken, mask, 0)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    ce = -jnp.sum(one_hot * log_probs, axis=-1)
    ce = jnp.sum(ce * pad_mask, axis=-1) / jnp.sum(pad_mask, axis=-1)
    return jnp.mean(ce)


LOSSES = {"cross_entropy_loss": cross_entropy_loss}
