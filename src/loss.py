import haiku as hk
import jax
import jax.numpy as jnp


@hk.without_apply_rng
@hk.transform
def cross_entropy_loss(x, y, padToken, forward):
    """
    Cross-entropy loss
    """
    # Loss is the average negative log-likelihood per token (excluding the first token)
    logits = forward(x).unembedded_output
    log_probs = jax.nn.log_softmax(logits)
    one_hot_targets = jax.nn.one_hot(y, logits.shape[-1])
    log_likelihood = jnp.sum(one_hot_targets * log_probs, axis=-1)
    # Mask the first token (BOS)
    mask = jnp.ones_like(log_likelihood)
    mask = mask.at[:, 0].set(0.0)
    # Mask the padding tokens
    padMask = jnp.where(x != padToken, mask, 0.0)
    # Return the average negative log-likelihood per token
    return -jnp.mean(log_likelihood * padMask) / jnp.sum(padMask)


@hk.without_apply_rng
@hk.transform
def cross_entropy_loss_with_perfect_sequence(x, y, padToken, forward):
    """
    Cross-entropy loss + zero loss on perfect sequence
    """
    logits = forward(x).unembedded_output
    log_probs = jax.nn.log_softmax(logits)
    one_hot_targets = jax.nn.one_hot(y, logits.shape[-1])

    # Mask the first token (BOS) and padding tokens
    mask = jnp.ones_like(y)
    mask = mask.at[:, 0].set(0)
    # Mask the padding tokens
    pad_mask = jnp.where(x != padToken, mask, 0)

    # Compute cross-entropy loss for each function
    ce_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    ce_loss = jnp.sum(ce_loss * pad_mask, axis=-1) / jnp.sum(pad_mask, axis=-1)

    # Check if the predictions are perfect
    # Mask the prediction and target for computing the prefect mask
    predictions = jnp.argmax(logits, axis=-1)
    predictions = predictions * pad_mask
    y = y * pad_mask
    perfect_mask = jnp.all(predictions == y, axis=-1)

    # If the predictions are perfect, set the loss to 0.0
    loss = jnp.where(perfect_mask, 0.0, ce_loss)
    loss = jnp.mean(loss)

    return loss


# Cross-entropy loss + smoothed accuracy
@hk.without_apply_rng
@hk.transform
def cross_entropy_loss_smoothed_accuracy(x, y, padToken, forward, accuracy_weight=0.9):
    """
    Cross-entropy loss + zero loss on perfect sequence
    """
    logits = forward(x).unembedded_output
    log_probs = jax.nn.log_softmax(logits)
    one_hot_targets = jax.nn.one_hot(y, logits.shape[-1])

    # Mask the first token (BOS) and padding tokens
    mask = jnp.ones_like(y)
    mask = mask.at[:, 0].set(0)
    # Mask the padding tokens
    pad_mask = jnp.where(x != padToken, mask, 0)

    # Compute cross-entropy loss for each function
    ce_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    ce_loss = jnp.sum(ce_loss * pad_mask, axis=-1) / jnp.sum(pad_mask, axis=-1)
    jax.debug.print("ce_loss: {x}", x=ce_loss)

    # Check if the predictions are perfect
    # Mask the prediction and target for computing the prefect mask
    predictions = jnp.argmax(logits, axis=-1)
    predictions = predictions * pad_mask
    y = y * pad_mask
    perfect_mask = jnp.all(predictions == y, axis=-1)

    # If the predictions are perfect, set the loss to ce_loss - accuracy_weight * ce_loss
    loss = jnp.where(perfect_mask, ce_loss - accuracy_weight * ce_loss, ce_loss)
    loss = jnp.mean(loss)

    return loss
