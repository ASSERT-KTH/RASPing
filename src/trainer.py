import jax
import optax
import haiku as hk
import numpy as np
import jax.numpy as jnp

import tqdm
import matplotlib.pyplot as plt
from typing import NamedTuple


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    step: jax.Array


class Trainer:

    def __init__(
        self,
        model,
        params,
        X_train,
        Y_train,
        loss_fn,
        n_epochs=1,
        batch_size=8,
        lr=0.0001,
        plot=False,
        X_val=None,
        Y_val=None,
        valCount=0,
        valStep=0,
    ):
        self.model = model
        self.params = params
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

        metrics = []
        validations = []

        # Set up early stopping
        if self.valCount:
            if self.X_val is None or self.Y_val is None:
                print("Error: X_val and Y_val not provided")
                return -1
        higherVal = 0
        latestVal = np.inf

        stoppedTraining = False
        for epoch in tqdm.trange(self.n_epochs):
            for i in range(0, len(self.X_train), self.batch_size):
                x = self.X_train[i : i + self.batch_size]
                y = self.Y_train[i : i + self.batch_size]
                self.state, metric = self.jit_update(self.state, x, y, padToken)

            metrics.append(metric)

            # Early stopping
            # TODO: fix calling of loss function
            if self.valCount:
                x = self.X_val
                y = self.Y_val
                newVal = self.jit_val_loss(
                    self.state.params, x, y, padToken
                )  # Validation loss

                if newVal > latestVal:
                    higherVal += 1
                    if higherVal == self.valCount:
                        print(
                            "Stopped training after", epoch, "epochs by early stopping"
                        )
                        stoppedTraining = True
                        break
                else:
                    higherVal = 0
                latestVal = newVal

            if self.valStep and epoch % self.valStep == 0:
                val = self.jit_val_accuracy(
                    self.state.params, self.X_val, self.Y_val, padToken
                )
                validations.append(val)

            if stoppedTraining:
                break

        if self.plot:
            # plot the loss values
            plt.plot([m["step"] for m in metrics], [m["loss"] for m in metrics])
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.show()

            if self.valStep:
                # plot the validation accuracies
                plt.plot(
                    np.linspace(0, self.n_epochs, len(validations)),
                    [m for m in validations],
                )
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.title("Validation Accuracy")
                plt.show()

        self.model.params = self.state.params
        if self.valStep:
            return metrics, validations
