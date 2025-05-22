import jax
import jax.numpy as jnp

def exact_match_reward(y_pred, y_true, pad_mask):
    """
    Gives +1 reward for exact matches, 0 otherwise.
    
    Args:
        y_pred: Predicted token indices
        y_true: Ground truth token indices
        pad_mask: Mask for padding tokens (1 for real tokens, 0 for padding)
        
    Returns:
        Reward scalar for each sequence in the batch
    """
    matches = jnp.all((y_pred == y_true) * pad_mask, axis=-1)
    return jnp.where(matches, 1.0, 0.0)

def token_match_reward(y_pred, y_true, pad_mask):
    """
    Gives reward based on proportion of correct tokens.
    
    Args:
        y_pred: Predicted token indices
        y_true: Ground truth token indices
        pad_mask: Mask for padding tokens (1 for real tokens, 0 for padding)
        
    Returns:
        Reward scalar for each sequence in the batch (between 0 and 1)
    """
    matches = jnp.sum((y_pred == y_true) * pad_mask, axis=-1)
    total = jnp.sum(pad_mask, axis=-1)
    return matches / total

def binary_token_reward(y_pred, y_true, pad_mask, reward_correct=1.0, penalty_incorrect=-0.1):
    """
    Gives specified reward for each correct token and penalty for each incorrect token.
    
    Args:
        y_pred: Predicted token indices
        y_true: Ground truth token indices
        pad_mask: Mask for padding tokens (1 for real tokens, 0 for padding)
        reward_correct: Reward value for correct tokens
        penalty_incorrect: Penalty value for incorrect tokens
        
    Returns:
        Total reward for each sequence in the batch
    """
    correct_tokens = (y_pred == y_true) * pad_mask
    incorrect_tokens = (y_pred != y_true) * pad_mask
    
    rewards = (correct_tokens * reward_correct) + (incorrect_tokens * penalty_incorrect)
    return jnp.sum(rewards, axis=-1)

def weighted_position_reward(y_pred, y_true, pad_mask, decay=0.9):
    """
    Position-weighted reward that gives higher importance to earlier tokens.
    
    Args:
        y_pred: Predicted token indices
        y_true: Ground truth token indices
        pad_mask: Mask for padding tokens (1 for real tokens, 0 for padding)
        decay: Factor by which rewards decay for each position
        
    Returns:
        Total reward for each sequence in the batch
    """
    seq_length = pad_mask.shape[-1]
    position_weights = jnp.power(decay, jnp.arange(seq_length))
    match_rewards = (y_pred == y_true) * pad_mask * position_weights
    return jnp.sum(match_rewards, axis=-1) 