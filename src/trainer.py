import os
import jax
import optax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import wandb
import pickle

import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from typing import NamedTuple, Optional, Any
from .early_stopping import EarlyStopping


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    step: jax.Array


class Trainer:

    def __init__(
        self,
        model,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        loss_fn: Any,
        n_epochs: int = 1,
        batch_size: int = 8,
        lr: float = 0.0001,
        plot: bool = False,
        X_val: Optional[jnp.ndarray] = None,
        Y_val: Optional[jnp.ndarray] = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        valStep: int = 0,
        output_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        from_pretrained: Optional[str] = None,
        store_trajectory: bool = False,
        trajectory_store_interval: int = 10,
    ):
        self.model = model.model
        self.params = model.model.params
        
        # Load pretrained weights if specified
        if from_pretrained is not None:
            success = model.load_model(from_pretrained)
            if success:
                self.params = model.model.params
                print(f"Successfully loaded pretrained model from {from_pretrained}")
            else:
                print(f"Warning: Failed to load pretrained model from {from_pretrained}")
                
        self.X_train = X_train
        self.Y_train = Y_train
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.plot = plot
        self.X_val = X_val
        self.Y_val = Y_val
        self.early_stopper = (
            EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode="min",
            )
            if early_stopping_patience > 0
            else None
        )
        self.valStep = valStep
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_wandb = use_wandb
        self.store_trajectory = store_trajectory
        self.trajectory_store_interval = trajectory_store_interval
        self.trajectory = [] # List to store (step, params, loss) tuples

        if self.use_wandb:
            wandb.init(
                project=wandb_project or "RASPing",
                name=wandb_name,
                config={
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "val_step": valStep,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                },
            )

        self.init()

    def init(self):
        ## Optimiser
        def optimiser(lr) -> optax.GradientTransformation:
            return optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(lr),
            )

        self.optimiser = optimiser

        initial_opt_state = optimiser(self.lr).init(self.params)
        self.state = TrainingState(
            params=self.params,
            opt_state=initial_opt_state,
            step=jnp.array(0),
        )

        ## Forward pass
        def forward(x):
            compiled_model = self.model.get_compiled_model()
            compiled_model.use_unembed_argmax = False
            compiled_model.pad_token = self.model.input_encoder.encoding_map[
                "compiler_pad"
            ]
            return compiled_model(x, use_dropout=False)

        _loss_fn_apply = jax.tree_util.Partial(self.loss_fn.apply, forward=forward)
        _value_and_grad = jax.value_and_grad(_loss_fn_apply)

        ## Step function
        def update(state: TrainingState, x, y, padToken):
            loss, grads = _value_and_grad(state.params, x, y, padToken)
            updates, opt_state = self.optimiser(self.lr).update(grads, state.opt_state)
            params = optax.apply_updates(state.params, updates)
            metrics = {"step": state.step, "loss": loss}
            return TrainingState(params, opt_state, step=state.step + 1), metrics

        self.jit_update = jax.jit(update)

        ## Validation loss
        def _val_loss_fn(params, x, y, padToken):
            return _loss_fn_apply(params, x, y, padToken)

        self.jit_val_loss = jax.jit(_val_loss_fn)

        ## Validation accuracy
        @hk.without_apply_rng
        @hk.transform
        def _val_accuracy(x, y, padToken, forward):
            logits = forward(jnp.array(x)).unembedded_output
            pred = jnp.argmax(logits, axis=-1)

            # Mask the first token (BOS)
            mask = jnp.ones_like(x)
            mask = mask.at[:, 0].set(0)
            # Mask the padding tokens
            padMask = jnp.where(x != padToken, mask, 0)
            val = jnp.mean(
                jnp.all(pred * padMask == y * padMask, axis=[-1]).astype(float)
            )
            return val

        _val_accuracy = jax.tree_util.Partial(_val_accuracy.apply, forward=forward)
        self.jit_val_accuracy = jax.jit(_val_accuracy)

    def train(self):
        padToken = self.model.input_encoder.encoding_map["compiler_pad"]

        # Initialize metrics arrays to store training and validation metrics
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # Calculate initial loss and store initial state if trajectory storage is enabled
        if self.store_trajectory and self.output_dir:
            # Calculate initial loss on the first batch of training data
            initial_loss = self.jit_val_loss(
                self.state.params, 
                self.X_train[:self.batch_size], 
                self.Y_train[:self.batch_size], 
                padToken
            )
            self.trajectory.append((0, jax.device_get(self.state.params), initial_loss))

        # Set up early stopping validation requirements
        if self.early_stopper is not None:
            if self.X_val is None or self.Y_val is None:
                print(
                    "Error: X_val and Y_val must be provided when using early stopping"
                )
                return -1

        stoppedTraining = False
        for epoch in tqdm.trange(self.n_epochs):
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(self.X_train), self.batch_size):
                x = self.X_train[i : i + self.batch_size]
                y = self.Y_train[i : i + self.batch_size]
                self.state, metric = self.jit_update(self.state, x, y, padToken)
                epoch_loss += metric["loss"]
                n_batches += 1
                
                # Store state at intervals if trajectory storage is enabled
                current_step = int(self.state.step)
                if self.store_trajectory and self.output_dir and current_step % self.trajectory_store_interval == 0:
                    # Store step, current (updated) params, and the loss calculated with these params
                    self.trajectory.append((current_step, jax.device_get(self.state.params), metric['loss']))

            avg_epoch_loss = epoch_loss / n_batches
            train_losses.append(avg_epoch_loss)

            # Calculate training accuracy
            train_acc = self.jit_val_accuracy(
                self.state.params, self.X_train, self.Y_train, padToken
            )
            train_accs.append(train_acc)

            # Validation and early stopping
            if self.early_stopper is not None or self.valStep:
                val_metrics = {}
                if self.X_val is not None and self.Y_val is not None:
                    val_loss = self.jit_val_loss(
                        self.state.params, self.X_val, self.Y_val, padToken
                    )
                    val_losses.append(val_loss)
                    val_metrics["val_loss"] = val_loss

                    if self.valStep and epoch % self.valStep == 0:
                        val_acc = self.jit_val_accuracy(
                            self.state.params, self.X_val, self.Y_val, padToken
                        )
                        val_accs.append(val_acc)
                        val_metrics["val_accuracy"] = val_acc

                    # Early stopping check with current parameters
                    if self.early_stopper is not None:
                        if self.early_stopper(val_loss, self.state.params):
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                            stoppedTraining = True

                # Log metrics to wandb
                if self.use_wandb:
                    wandb_metrics = {
                        "epoch": epoch,
                        "train_loss": avg_epoch_loss,
                        "train_accuracy": train_acc,
                        **val_metrics,
                    }
                    wandb.log(wandb_metrics)

            if stoppedTraining:
                # Restore best model parameters
                if (
                    self.early_stopper is not None
                    and self.early_stopper.best_params is not None
                ):
                    print("Restoring best model parameters...")
                    self.state = TrainingState(
                        params=self.early_stopper.best_params,
                        opt_state=self.state.opt_state,
                        step=self.state.step,
                    )
                break

        # Calculate final loss and store final state if trajectory storage is enabled
        if self.store_trajectory and self.output_dir:
            final_step = int(self.state.step)
            final_params = jax.device_get(self.state.params)
            # Ensure final state isn't identical to the last stored state due to interval
            if not self.trajectory or self.trajectory[-1][0] != final_step:
                # Recalculate loss for the final parameters on a training batch
                padToken = self.model.input_encoder.encoding_map["compiler_pad"]
                final_loss = self.jit_val_loss(
                    final_params, 
                    self.X_train[:self.batch_size], 
                    self.Y_train[:self.batch_size], 
                    padToken
                )
                self.trajectory.append((final_step, final_params, final_loss))

        if self.plot:
            # Plot training and validation loss
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label="Training Loss")
            if val_losses:
                plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()

            # Plot training and validation accuracy
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label="Training Accuracy")
            if val_accs:
                plt.plot(val_accs, label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.show()

        self.model.params = self.state.params

        if self.output_dir:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            self.save_metrics(train_losses, train_accs, val_losses, val_accs)
            self.save_model()
            if self.store_trajectory:
                self.save_trajectory()

        if self.use_wandb:
            wandb.finish()

        return {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        }

    def save_metrics(self, train_losses, train_accs, val_losses, val_accs):
        np.save(self.output_dir / "train_losses.npy", train_losses)
        np.save(self.output_dir / "train_accs.npy", train_accs)
        if val_losses:
            np.save(self.output_dir / "val_losses.npy", val_losses)
        if val_accs:
            np.save(self.output_dir / "val_accs.npy", val_accs)

    def save_model(self):
        np.save(self.output_dir / "model.npy", self.state.params)

    def save_trajectory(self):
        """Save the training trajectory to a pickle file."""
        if not self.output_dir:
            print("Warning: Output directory not set, cannot save trajectory.")
            return
        trajectory_path = self.output_dir / "trajectory.pkl"
        try:
            with open(trajectory_path, "wb") as f:
                pickle.dump(self.trajectory, f)
            print(f"Trajectory saved to {trajectory_path}")
        except Exception as e:
            print(f"Error saving trajectory to {trajectory_path}: {e}")

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
