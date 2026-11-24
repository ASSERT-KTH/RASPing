"""Gradient-based training of a compiled model: Adam + global-norm clipping + early stopping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

import tqdm

from rasping.model import Model


class EarlyStopping:
    """Stop when the monitored value has not improved by `min_delta` for `patience` checks.

    Remembers the parameters (and step) of the best value seen.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode {mode} is unknown, should be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta) * (1 if mode == "min" else -1)
        self.monitor_op = np.less if mode == "min" else np.greater
        self.reset()

    def reset(self) -> None:
        self.counter = 0
        self.best_value = np.inf if self.mode == "min" else -np.inf
        self.best_params = None
        self.best_step = None
        self.should_stop = False

    def __call__(self, value: float, params=None, step: int | None = None) -> bool:
        """Record `value`; return True if training should stop."""
        if self.monitor_op(value + self.min_delta, self.best_value):
            self.best_value = value
            if params is not None:
                self.best_params = params
            if step is not None:
                self.best_step = step
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    step: jax.Array


@dataclass
class TrainResult:
    train_losses: list = field(default_factory=list)  # per epoch (mean over batches)
    train_accs: list = field(default_factory=list)  # per epoch, full train set
    val_losses: list = field(default_factory=list)  # per epoch
    val_accs: list = field(default_factory=list)  # every `val_every` epochs (epoch 0, 10, ...)
    # test accuracy of the parameters early stopping would return at that point (best
    # validation loss so far), at the same epochs as val_accs; only if X_test is given
    selected_test_accs: list = field(default_factory=list)
    stopped_early: bool = False
    diverged: bool = False  # training loss became non-finite; params are the last finite ones
    best_step: Optional[int] = None
    final_step: int = 0
    trajectory: list = field(default_factory=list)  # (step, params, val_loss) if requested
    epoch_end_times: list = field(default_factory=list)  # seconds since train() started

    def history(self) -> dict:
        keys = ["train_losses", "train_accs", "val_losses", "val_accs"]
        if self.selected_test_accs:
            keys.append("selected_test_accs")
        return {k: [float(v) for v in getattr(self, k)] for k in keys}


class Trainer:
    """Supervised training of a `Model`'s weights on an encoded specification.

    The training set is reshuffled at the start of every epoch with a NumPy generator
    seeded by `seed`, so a run is reproducible given its seed and different seeds give
    different batch orders. (v1 took minibatches in the same fixed order every epoch, which
    made compiled-init runs deterministic: FINDINGS F36.) With early stopping, the
    validation loss is checked every epoch and the best parameters are restored.
    """

    def __init__(
        self,
        model: Model,
        X_train, Y_train,
        loss_fn: Any,
        n_epochs: int = 1,
        batch_size: int = 8,
        lr: float = 1e-4,
        X_val=None, Y_val=None,
        X_test=None, Y_test=None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        val_every: int = 0,
        grad_clip_norm: float = 1.0,
        store_trajectory: bool = False,
        trajectory_store_interval: int = 10,
        wandb_run: Any = None,
        progress: bool = True,
        seed: int = 0,
    ):
        self.model = model
        # Own stream: `seed` also seeds subsampling and re-initialisation in rasping.repair.
        self._rng = np.random.default_rng([seed, 1])
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val
        self.X_test, self.Y_test = X_test, Y_test  # monitoring only, never used for decisions
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.val_every = val_every
        self.store_trajectory = store_trajectory
        self.trajectory_store_interval = trajectory_store_interval
        self.wandb_run = wandb_run
        self.progress = progress
        self.early_stopper = (
            EarlyStopping(early_stopping_patience, early_stopping_min_delta, "min")
            if early_stopping_patience > 0 else None
        )
        if self.early_stopper is not None and (X_val is None or Y_val is None):
            raise ValueError("X_val and Y_val are required for early stopping")

        optimiser = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr))
        self.state = TrainingState(model.params, optimiser.init(model.params), jnp.array(0))

        loss_apply = jax.tree_util.Partial(loss_fn.apply, forward=model._forward)
        value_and_grad = jax.value_and_grad(loss_apply)

        def update(state: TrainingState, x, y, pad):
            loss, grads = value_and_grad(state.params, x, y, pad)
            updates, opt_state = optimiser.update(grads, state.opt_state)
            params = optax.apply_updates(state.params, updates)
            return TrainingState(params, opt_state, state.step + 1), loss

        self._update = jax.jit(update)
        self._loss = jax.jit(loss_apply)

    def _val_loss(self, params):
        return self._loss(params, self.X_val, self.Y_val, self.model.pad_token)

    def train(self) -> TrainResult:
        pad = self.model.pad_token
        res = TrainResult()
        t0 = time.time()
        if self.store_trajectory:
            res.trajectory.append((0, jax.device_get(self.state.params), self._val_loss(self.state.params)))

        prev_params = self.state.params
        epochs = tqdm.trange(self.n_epochs) if self.progress else range(self.n_epochs)
        for epoch in epochs:
            epoch_loss, n_batches = 0.0, 0
            perm = self._rng.permutation(len(self.X_train))
            X_epoch, Y_epoch = self.X_train[perm], self.Y_train[perm]
            for i in range(0, len(X_epoch), self.batch_size):
                x = X_epoch[i:i + self.batch_size]
                y = Y_epoch[i:i + self.batch_size]
                self.state, loss = self._update(self.state, x, y, pad)
                epoch_loss += loss
                n_batches += 1
                step = int(self.state.step)
                if self.store_trajectory and step % self.trajectory_store_interval == 0:
                    res.trajectory.append((step, jax.device_get(self.state.params),
                                           self._val_loss(self.state.params)))

            if not np.isfinite(float(epoch_loss)):
                res.diverged = True
                self.state = self.state._replace(params=prev_params)
                break
            prev_params = self.state.params
            res.train_losses.append(epoch_loss / n_batches)
            train_acc = self.model.accuracy(self.X_train, self.Y_train, self.state.params)
            res.train_accs.append(train_acc)

            log = {"epoch": epoch, "train_loss": res.train_losses[-1], "train_accuracy": train_acc}
            stop = False
            if (self.early_stopper is not None or self.val_every) and self.X_val is not None:
                val_loss = self._val_loss(self.state.params)
                res.val_losses.append(val_loss)
                log["val_loss"] = val_loss
                if self.val_every and epoch % self.val_every == 0:
                    val_acc = self.model.accuracy(self.X_val, self.Y_val, self.state.params)
                    res.val_accs.append(val_acc)
                    log["val_accuracy"] = val_acc
                if self.early_stopper is not None:
                    stop = self.early_stopper(val_loss, self.state.params, int(self.state.step))
                if self.val_every and epoch % self.val_every == 0 and self.X_test is not None:
                    es = self.early_stopper
                    selected = es.best_params if es is not None and es.best_params is not None \
                        else self.state.params
                    res.selected_test_accs.append(self.model.accuracy(self.X_test, self.Y_test, selected))
            res.epoch_end_times.append(time.time() - t0)
            if self.wandb_run is not None:
                self.wandb_run.log({k: float(v) for k, v in log.items()}, step=int(self.state.step))

            if stop:
                res.stopped_early = True
                res.best_step = self.early_stopper.best_step
                if self.early_stopper.best_params is not None:
                    self.state = self.state._replace(params=self.early_stopper.best_params)
                    if self.store_trajectory and res.best_step is not None:
                        res.trajectory = [t for t in res.trajectory if t[0] <= res.best_step]
                break

        res.final_step = int(self.state.step)
        if self.store_trajectory and (not res.trajectory or res.trajectory[-1][0] != res.final_step):
            params = jax.device_get(self.state.params)
            res.trajectory.append((res.final_step, params, self._val_loss(params)))

        self.model.assembled.params = self.state.params
        return res
