import sys
import os
import click

from pathlib import Path

# TODO: this is a hack, change this for editable install
module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.trainer import Trainer
from src.functions import load_dataset, encodeAndPadData
from src.loss import cross_entropy_loss
from experiments.mutation.load_mutations import load_buggy_models


def train_mutated_model(
    program_name: str,
    job_id: str,
    max_len: int = 10,
    n_epochs: int = 50000,
    batch_size: int = 256,
    learning_rate: float = 1e-04,
    output_dir: str = None,
):
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

    # Train the model and get metrics
    trainer = Trainer(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        loss_fn=cross_entropy_loss,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        plot=False,
        X_val=X_val,
        Y_val=Y_val,
        valCount=0,
        valStep=10,
        output_dir=output_dir,
        use_wandb=True,
        wandb_project="dpr-mutation-training",
        wandb_name=f"{program_name}_{job_id}",
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
@click.option("--output_dir", type=str, help="Directory to save training outputs")
def run_test(
    program_name, job_id, max_len, n_epochs, batch_size, learning_rate, output_dir
):
    print(f"Training mutated model {program_name} (job {job_id})...")
    if not output_dir:
        output_dir = f"saved_data/{program_name}/job_{job_id}/"
    train_mutated_model(
        program_name=program_name,
        job_id=job_id,
        max_len=max_len,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    run_test()
