import os
import jax
import optax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import wandb

import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from typing import NamedTuple, Optional, Any, Callable

from .rewards import exact_match_reward


class REINFORCEState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    step: jax.Array
    prng_key: jax.Array


class REINFORCETrainer:
    """
    Implements the REINFORCE (Monte Carlo policy gradient) algorithm for training models.
    
    This trainer uses reinforcement learning to optimize model parameters by:
    1. Sampling actions (tokens) from the model's probability distribution
    2. Computing rewards for the complete sequences
    3. Updating parameters using policy gradients weighted by rewards
    """

    def __init__(
        self,
        model,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,  # Only used for computing rewards
        reward_fn: Callable = exact_match_reward,
        n_epochs: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.99,  # Discount factor
        temperature: float = 1.0,  # Sampling temperature
        entropy_coef: float = 0.01,  # Coefficient for entropy regularization
        baseline: str = "none",  # Options: "none", "mean", "value"
        output_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: int = 42,
        from_pretrained: Optional[str] = None,
    ):      
        # Load pretrained weights if specified
        if from_pretrained is not None:
            success = self.load_saved_model(model, from_pretrained)
            if success:
                print(f"Successfully loaded pretrained model from {from_pretrained}")
            else:
                print(f"Warning: Failed to load pretrained model from {from_pretrained}")

        # Set model and params
        self.model = model.model
        self.params = model.model.params

        self.X_train = X_train
        self.Y_train = Y_train
        self.reward_fn = reward_fn
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.temperature = temperature
        self.entropy_coef = entropy_coef
        self.baseline = baseline
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_wandb = use_wandb
        self.prng_key = jax.random.key(seed)

        if self.use_wandb:
            wandb.init(
                project=wandb_project or "RASPing-RL",
                name=wandb_name,
                config={
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "gamma": gamma,
                    "temperature": temperature,
                    "entropy_coef": entropy_coef,
                    "baseline": baseline,
                    "reward_fn": reward_fn.__name__,
                },
            )

        self.init()

    def init(self):
        # Set up optimizer
        def optimizer(lr) -> optax.GradientTransformation:
            return optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(lr),
            )

        self.optimizer = optimizer

        initial_opt_state = optimizer(self.lr).init(self.params)
        self.state = REINFORCEState(
            params=self.params,
            opt_state=initial_opt_state,
            step=jnp.array(0),
            prng_key=self.prng_key,
        )

        # Set up forward pass
        def forward(x):
            compiled_model = self.model.get_compiled_model()
            compiled_model.use_unembed_argmax = False
            compiled_model.pad_token = self.model.input_encoder.encoding_map[
                "compiler_pad"
            ]
            return compiled_model(x, use_dropout=False)

        self.forward_fn = forward
        self.pad_token = self.model.input_encoder.encoding_map["compiler_pad"]

        # Function to sample actions from logits
        def sample_actions(params, x, temperature, key):
            """Sample actions from model's policy distribution"""
            logits = hk.without_apply_rng(hk.transform(forward)).apply(params, x).unembedded_output
            
            # Scale logits by temperature
            scaled_logits = logits / temperature
            
            # Get probabilities
            probs = jax.nn.softmax(scaled_logits, axis=-1)
            
            # Sample from the distribution (for each token position)
            keys = jax.random.split(key, x.shape[1] + 1)
            
            # Collect samples for each position
            samples = []
            for i in range(x.shape[1]):
                sample = jax.random.categorical(keys[i], probs[:, i, :])
                samples.append(sample)
            
            return jnp.stack(samples, axis=1), probs, keys[-1]

        self.sample_actions = jax.jit(sample_actions)

        # REINFORCE loss function
        def reinforce_loss(params, x, sampled_actions, rewards, key, temperature):
            """Policy gradient loss for REINFORCE"""
            # Get model outputs
            logits = hk.without_apply_rng(hk.transform(forward)).apply(params, x).unembedded_output
            
            # Get log probabilities
            log_probs = jax.nn.log_softmax(logits / temperature, axis=-1)
            
            # Create masks
            mask = jnp.ones_like(x)
            mask = mask.at[:, 0].set(0)  # Mask BOS token
            pad_mask = jnp.where(x != self.pad_token, mask, 0)
            
            # For each token position, get the log probability of the sampled action
            # Fixed version to handle indexing properly
            batch_size = x.shape[0]
            seq_length = x.shape[1]
            
            # Create a new tensor to store the log probabilities of sampled actions
            action_log_probs = jnp.zeros((batch_size, seq_length))
            
            # Extract log probabilities for each sampled action in each position
            batch_indices = jnp.arange(batch_size)
            
            for i in range(seq_length):
                # For position i, get log probs for the actions selected
                # This uses advanced indexing to get the right values
                action_indices = sampled_actions[:, i]
                action_log_probs = action_log_probs.at[:, i].set(
                    log_probs[batch_indices, i, action_indices]
                )
            
            # Mask out padded positions
            masked_log_probs = action_log_probs * pad_mask
            
            # Compute policy gradient loss
            pg_loss = -jnp.sum(masked_log_probs * rewards[:, None], axis=1)
            
            # Add entropy regularization
            entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
            entropy = jnp.sum(entropy * pad_mask, axis=1)
            
            # Combine losses
            loss = jnp.mean(pg_loss - self.entropy_coef * entropy)
            
            return loss

        self.reinforce_loss_fn = reinforce_loss

        # Update function for REINFORCE
        def update(state, x, y, temperature):
            """Update model parameters using REINFORCE algorithm"""
            # Split the PRNG key
            key, sample_key, next_key = jax.random.split(state.prng_key, 3)
            
            # Sample actions from policy
            sampled_actions, probs, new_key = self.sample_actions(
                state.params, x, temperature, sample_key
            )
            
            # Compute rewards for sampled actions
            pad_mask = jnp.where(x != self.pad_token, jnp.ones_like(x), 0)
            pad_mask = pad_mask.at[:, 0].set(0)  # Mask BOS token
            
            rewards = self.reward_fn(sampled_actions, y, pad_mask)
            
            # Apply baseline if specified
            if self.baseline == "mean":
                rewards = rewards - jnp.mean(rewards)
            
            # Compute loss and gradients
            loss_fn = lambda p: self.reinforce_loss_fn(p, x, sampled_actions, rewards, key, temperature)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            
            # Update parameters
            updates, opt_state = self.optimizer(self.lr).update(grads, state.opt_state)
            params = optax.apply_updates(state.params, updates)
            
            # Compute metrics
            match_count = jnp.sum(rewards > 0)
            match_rate = match_count / rewards.shape[0]
            
            metrics = {
                "loss": loss, 
                "reward_mean": jnp.mean(rewards),
                "reward_std": jnp.std(rewards),
                "match_rate": match_rate,
            }
            
            # Return updated state and metrics
            new_state = REINFORCEState(
                params=params,
                opt_state=opt_state,
                step=state.step + 1,
                prng_key=next_key,
            )
            
            return new_state, metrics, sampled_actions, rewards

        self.update_fn = jax.jit(update)

        # Evaluation function
        def evaluate(params, x, y):
            """Evaluate model performance using greedy sampling (argmax)"""
            logits = hk.without_apply_rng(hk.transform(forward)).apply(params, x).unembedded_output
            predictions = jnp.argmax(logits, axis=-1)
            
            # Create mask for valid tokens
            mask = jnp.ones_like(x)
            mask = mask.at[:, 0].set(0)  # Mask BOS token
            pad_mask = jnp.where(x != self.pad_token, mask, 0)
            
            # Calculate exact match accuracy
            accuracy = jnp.mean(
                jnp.all(predictions * pad_mask == y * pad_mask, axis=-1).astype(float)
            )
            
            return accuracy, predictions

        self.evaluate_fn = jax.jit(evaluate)

    def train(self):
        """Train the model using REINFORCE algorithm"""
        # Initialize metrics tracking
        train_rewards = []
        train_losses = []
        train_accuracies = []
        best_accuracy = 0.0
        best_params = None

        # Training loop
        for epoch in tqdm.trange(self.n_epochs):
            epoch_rewards = []
            epoch_losses = []
            epoch_matches = []
            
            # Process in batches
            for i in range(0, len(self.X_train), self.batch_size):
                x = self.X_train[i:i+self.batch_size]
                y = self.Y_train[i:i+self.batch_size]
                
                # Update model using REINFORCE
                self.state, metrics, actions, rewards = self.update_fn(
                    self.state, x, y, self.temperature
                )
                
                # Record metrics
                epoch_rewards.append(metrics["reward_mean"])
                epoch_losses.append(metrics["loss"])
                epoch_matches.append(metrics["match_rate"])
            
            # Compute epoch averages
            avg_reward = np.mean(epoch_rewards)
            avg_loss = np.mean(epoch_losses)
            avg_match_rate = np.mean(epoch_matches)
            
            train_rewards.append(avg_reward)
            train_losses.append(avg_loss)
            
            # Evaluate model performance (with greedy sampling)
            accuracy, _ = self.evaluate_fn(self.state.params, self.X_train, self.Y_train)
            train_accuracies.append(accuracy)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = self.state.params
                if self.output_dir:
                    self.save_model(best=True)
            
            # Log metrics
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_reward": avg_reward,
                    "train_loss": avg_loss,
                    "train_match_rate": avg_match_rate,
                    "train_accuracy": accuracy,
                })
            
            # Occasionally print progress
            if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                print(f"Epoch {epoch}: Reward={avg_reward:.4f}, "
                      f"Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Save final model
        if self.output_dir:
            self.save_model()
            self.save_metrics(train_rewards, train_losses, train_accuracies)
        
        # Update model with best parameters
        if best_params is not None:
            self.model.params = best_params
        
        if self.use_wandb:
            wandb.finish()
        
        return {
            "train_rewards": train_rewards,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "best_accuracy": best_accuracy,
        }

    def save_model(self, best=False):
        """Save model parameters to file"""
        prefix = "best_" if best else ""
        file_path = self.output_dir / f"{prefix}model.npy"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        np.save(file_path, self.state.params)

    def save_metrics(self, rewards, losses, accuracies):
        """Save training metrics to file"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        np.save(self.output_dir / "train_rewards.npy", rewards)
        np.save(self.output_dir / "train_losses.npy", losses)
        np.save(self.output_dir / "train_accuracies.npy", accuracies)
        
    @staticmethod
    def load_saved_model(model, model_path):
        """
        Load a saved model from a path
        
        Args:
            model: A Model instance to load parameters into
            model_path: Path to the directory containing model.npy or path to the model.npy file
            
        Returns:
            True if loading was successful, False otherwise
        """
        return model.load_model(model_path) 