import haiku as hk
import jax
import jax.numpy as jnp


def _create_masks(x, y, padToken):
    """Create common masks used across loss functions"""
    # Base mask with first token (BOS) zeroed out
    mask = jnp.ones_like(y)
    mask = mask.at[:, 0].set(0)
    # Mask for padding tokens
    pad_mask = jnp.where(x != padToken, mask, 0)
    return mask, pad_mask


def _compute_base_ce_loss(logits, y, pad_mask):
    """Compute basic cross entropy loss with masking"""
    log_probs = jax.nn.log_softmax(logits)
    one_hot_targets = jax.nn.one_hot(y, logits.shape[-1])
    ce_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    ce_loss = jnp.sum(ce_loss * pad_mask, axis=-1) / jnp.sum(pad_mask, axis=-1)
    return ce_loss, log_probs, one_hot_targets


def _get_perfect_mask(logits, y, pad_mask):
    """Check if predictions match targets perfectly"""
    predictions = jnp.argmax(logits, axis=-1)
    predictions = predictions * pad_mask
    y_masked = y * pad_mask
    return jnp.all(predictions == y_masked, axis=-1)


@hk.without_apply_rng
@hk.transform
def cross_entropy_loss(x, y, padToken, forward):
    """Basic cross-entropy loss"""
    logits = forward(x).unembedded_output
    _, pad_mask = _create_masks(x, y, padToken)
    ce_loss, _, _ = _compute_base_ce_loss(logits, y, pad_mask)
    return jnp.mean(ce_loss)


@hk.without_apply_rng
@hk.transform
def cross_entropy_loss_with_perfect_sequence(x, y, padToken, forward):
    """Cross-entropy loss + zero loss on perfect sequence"""
    logits = forward(x).unembedded_output
    _, pad_mask = _create_masks(x, y, padToken)
    ce_loss, _, _ = _compute_base_ce_loss(logits, y, pad_mask)
    perfect_mask = _get_perfect_mask(logits, y, pad_mask)
    loss = jnp.where(perfect_mask, 0.0, ce_loss)
    return jnp.mean(loss)


@hk.without_apply_rng
@hk.transform
def cross_entropy_loss_smoothed_accuracy(x, y, padToken, forward, accuracy_weight=0.9):
    """Cross-entropy loss + smoothed zero loss on perfect sequence"""
    logits = forward(x).unembedded_output
    _, pad_mask = _create_masks(x, y, padToken)
    ce_loss, _, _ = _compute_base_ce_loss(logits, y, pad_mask)
    perfect_mask = _get_perfect_mask(logits, y, pad_mask)
    loss = jnp.where(perfect_mask, ce_loss - accuracy_weight * ce_loss, ce_loss)
    return jnp.mean(loss)
