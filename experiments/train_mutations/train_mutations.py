import sys
import os
import math
import click
import numpy as np

from pathlib import Path

# TODO: this is a hack, change this for editable install
module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.trainer import Trainer
from src.functions import load_dataset, encodeAndPadData, generateModel
from src.loss import (
    cross_entropy_loss,
    cross_entropy_loss_smoothed_accuracy,
    cross_entropy_loss_with_perfect_sequence,
)
from experiments.mutation.load_mutations import load_buggy_models


LOSS_FUNCTIONS = {
    "cross_entropy_loss": cross_entropy_loss,
    "cross_entropy_loss_smoothed_accuracy": cross_entropy_loss_smoothed_accuracy,
    "cross_entropy_loss_with_perfect_sequence": cross_entropy_loss_with_perfect_sequence,
}


def train_mutated_model(
    program_name: str,
    job_id: str,
    max_len: int = 10,
    n_epochs: int = 50000,
    batch_size: int = 256,
    learning_rate: float = 1e-04,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    output_dir: str = None,
    loss_fn_name: str = "cross_entropy_loss",
    store_trajectory: bool = False,
    trajectory_store_interval: int = 10,
    n_train_samples: int = None,
    n_val_samples: int = None,
    seed: int = 42,
    n_steps: int = None,
    random_init: bool = False,
    init_scheme: str = "xavier",
    random_init_source: str = "buggy",
):
    if loss_fn_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function {loss_fn_name} not found")
    loss_fn = LOSS_FUNCTIONS[loss_fn_name]

    if random_init:
        # Initialization-control baseline: randomize the weights instead of
        # starting from the buggy mutation's compiled checkpoint. Everything
        # else (data, optimizer, early stopping, evaluation) matches the
        # mutant-repair path exactly.
        #
        # random_init_source controls WHICH architecture is randomized:
        #   "ground-truth" compiles the correct program. This confounds the
        #     comparison, because most mutants do not share the ground-truth
        #     architecture, so the baseline is handed the right structure for
        #     free -- it measures architecture + initialization together.
        #   "buggy" (default) loads the mutant and randomizes that same
        #     architecture, isolating the value of the compiled starting point
        #     from the architecture it lives in. This is the paired control.
        if random_init_source == "buggy":
            if not job_id:
                raise ValueError("--random-init-source buggy requires --job_id")
            model = load_buggy_models(
                max_length=max_len, program_name=program_name, job_id=job_id
            )[job_id]
        else:
            model = generateModel(program_name, max_len)
        model.setJaxPRNGKey(seed)
        model.setRandomWeights(
            scheme=None if init_scheme == "unit-normal" else init_scheme
        )
    else:
        # Load the buggy model
        model = load_buggy_models(
            max_length=max_len, program_name=program_name, job_id=job_id
        )[job_id]

    # Load dataset
    program_name_key = program_name
    if program_name == "most_freq":
        program_name_key = "most-freq"
    elif program_name == "shuffle_dyck":
        program_name_key = "shuffle_dyck1"
    data_path = f"{Path(__file__).parent.resolve()}/../../data/"
    train_dataset = load_dataset(data_path, program_name_key, split_name="train")
    val_dataset = load_dataset(data_path, program_name_key, split_name="val")

    # Encode the dataset
    X_train, Y_train = encodeAndPadData(
        train_dataset, model.raspFunction, model.inputs, max_len
    )
    X_val, Y_val = encodeAndPadData(
        val_dataset, model.raspFunction, model.inputs, max_len
    )

    # Optionally subsample the training set
    if n_train_samples is not None and n_train_samples < len(X_train):
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(X_train), n_train_samples, replace=False)
        X_train, Y_train = X_train[indices], Y_train[indices]

    if n_val_samples is not None and n_val_samples < len(X_val):
        rng_val = np.random.default_rng(seed=seed)
        val_indices = rng_val.choice(len(X_val), n_val_samples, replace=False)
        X_val, Y_val = X_val[val_indices], Y_val[val_indices]

    # Convert n_steps to n_epochs if provided (equal compute budget across N values)
    if n_steps is not None:
        steps_per_epoch = max(1, len(X_train) // batch_size)
        n_epochs = math.ceil(n_steps / steps_per_epoch)

    # Build W&B run name and extra config
    wandb_name = f"{program_name}_{job_id}_{loss_fn_name}"
    if n_train_samples is not None:
        wandb_name += f"_n{n_train_samples}_s{seed}"
    wandb_extra_config = {"n_train_samples": n_train_samples or len(X_train), "seed": seed}

    # Train the model and get metrics
    trainer = Trainer(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        plot=False,
        X_val=X_val,
        Y_val=Y_val,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        valStep=10,
        output_dir=output_dir,
        use_wandb=True,
        wandb_project="dpr-mutation-training",
        wandb_name=wandb_name,
        store_trajectory=store_trajectory,
        trajectory_store_interval=trajectory_store_interval,
        wandb_extra_config=wandb_extra_config,
    )

    # Trainer will train the model log metrics, and save metrics and results to output_dir
    trainer.train()


@click.command()
@click.option(
    "--program_name", type=str, help="The name of the program to load buggy models from"
)
@click.option(
    "--job_id", type=str, help="The job ID for loading the specific buggy model"
)
@click.option("--max_len", type=int, default=10, help="Maximum sequence length")
@click.option("--n_epochs", type=int, default=50000, help="Number of training epochs")
@click.option("--batch_size", type=int, default=256, help="Training batch size")
@click.option("--learning_rate", type=float, default=1e-04, help="Learning rate")
@click.option(
    "--early_stopping_patience",
    type=int,
    default=50,
    help="Number of epochs to wait before early stopping",
)
@click.option(
    "--early_stopping_min_delta",
    type=float,
    default=1e-4,
    help="Minimum change in monitored value to qualify as an improvement",
)
@click.option("--output_dir", type=str, help="Directory to save training outputs")
@click.option(
    "--loss_fn_name",
    type=click.Choice(list(LOSS_FUNCTIONS.keys())),
    default="cross_entropy_loss",
    help="Loss function to use for training",
)
@click.option(
    "--store-trajectory/--no-store-trajectory",
    default=False,
    help="Store the training trajectory (parameter history)"
)
@click.option(
    "--trajectory-store-interval",
    type=int,
    default=10,
    help="Interval (in steps) for storing trajectory points"
)
@click.option("--n_train_samples", type=int, default=None,
              help="Number of training samples to use (None = use all)")
@click.option("--n_val_samples", type=int, default=None,
              help="Number of val samples to use (None = use all)")
@click.option("--seed", type=int, default=42,
              help="Random seed for training-set subsampling")
@click.option("--n_steps", type=int, default=None,
              help="Total gradient steps (overrides n_epochs; ensures equal compute across N values)")
@click.option(
    "--random-init/--no-random-init",
    default=False,
    help="Initialization control: compile the ground-truth program and randomize its weights "
         "(seeded by --seed) instead of loading a buggy mutation. --job_id is not required in "
         "this mode and defaults to seed{seed} if omitted.",
)
@click.option(
    "--random-init-source",
    type=click.Choice(["buggy", "ground-truth"]),
    default="buggy",
    help="With --random-init, which architecture to randomize. 'buggy' (default) randomizes the "
         "mutant's own compiled architecture, isolating initialization from architecture. "
         "'ground-truth' compiles the correct program instead, which also hands the baseline the "
         "correct structure and so confounds the two.",
)
@click.option(
    "--init-scheme",
    type=click.Choice(["lecun", "xavier", "kaiming", "unit-normal"]),
    default="xavier",
    help="Weight initialization used with --random-init. Fan-in scaled schemes: 'xavier' "
         "(default; Glorot & Bengio 2010, as in the original Transformer), 'lecun' (Haiku's "
         "hk.Linear default), 'kaiming' (He et al. 2015). 'unit-normal' reproduces the historical "
         "N(0,1) behaviour, which is NOT fan-in scaled and starts these models numerically broken.",
)
def run_test(
    program_name,
    job_id,
    max_len,
    n_epochs,
    batch_size,
    learning_rate,
    early_stopping_patience,
    early_stopping_min_delta,
    output_dir,
    loss_fn_name,
    store_trajectory,
    trajectory_store_interval,
    n_train_samples,
    n_val_samples,
    seed,
    n_steps,
    random_init,
    init_scheme,
    random_init_source,
):
    if random_init and not job_id:
        job_id = f"seed{seed}"

    print(f"Training {'random-init' if random_init else 'mutated'} model {program_name} "
          f"(job {job_id}) with {loss_fn_name}...")
    if not output_dir:
        output_dir = f"saved_data/{program_name}/{loss_fn_name}/job_{job_id}/"
    train_mutated_model(
        program_name=program_name,
        job_id=job_id,
        max_len=max_len,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        output_dir=output_dir,
        loss_fn_name=loss_fn_name,
        store_trajectory=store_trajectory,
        trajectory_store_interval=trajectory_store_interval,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        seed=seed,
        n_steps=n_steps,
        random_init=random_init,
        init_scheme=init_scheme,
        random_init_source=random_init_source,
    )


if __name__ == "__main__":
    run_test()
