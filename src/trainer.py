import os
import jax
import optax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import wandb

import tqdm
import matplotlib.pyplot as plt
from typing import NamedTuple, Optional, Any


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
        valCount: int = 0,
        valStep: int = 0,
        output_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
    ):
        self.model = model.model
        self.params = model.model.params
        self.X_train = X_train
        self.Y_train = Y_train
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.plot = plot
        self.X_val = X_val
        self.Y_val = Y_val
        self.valCount = valCount
        self.valStep = valStep
        self.output_dir = output_dir
        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(
                project=wandb_project or "RASPing",
                name=wandb_name,
                config={
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "val_step": valStep,
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

        # Set up early stopping
        if self.valCount:
            if self.X_val is None or self.Y_val is None:
                print("Error: X_val and Y_val not provided")
                return -1
        higherVal = 0
        latestVal = np.inf

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

            avg_epoch_loss = epoch_loss / n_batches
            train_losses.append(avg_epoch_loss)

            # Calculate training accuracy
            train_acc = self.jit_val_accuracy(
                self.state.params, self.X_train, self.Y_train, padToken
            )
            train_accs.append(train_acc)

            # Early stopping and validation
            if self.valCount or self.valStep:
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

                    # Early stopping check
                    if self.valCount:
                        if val_loss > latestVal:
                            higherVal += 1
                            if higherVal == self.valCount:
                                print(
                                    f"Stopped training after {epoch} epochs by early stopping"
                                )
                                stoppedTraining = True
                        else:
                            higherVal = 0
                        latestVal = val_loss

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
                break

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

        if self.use_wandb:
            wandb.finish()

        return {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        }

    def save_metrics(self, train_losses, train_accs, val_losses, val_accs):
        np.save(self.output_dir + "train_losses.npy", train_losses)
        np.save(self.output_dir + "train_accs.npy", train_accs)
        if val_losses:
            np.save(self.output_dir + "val_losses.npy", val_losses)
        if val_accs:
            np.save(self.output_dir + "val_accs.npy", val_accs)

    def save_model(self):
        np.save(self.output_dir + "model.npy", self.state.params)
