import sys
import os
import click

# TODO: this is a hack, change this for editable install
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.trainer import Trainer
from src.functions import load_dataset
from src.loss import cross_entropy_loss
from experiments.mutation.load_mutations import load_buggy_models

def train_mutated_model(model_name: str, program_name: str, job_id: int, max_len: int = 10, n_epochs: int = 50000, batch_size: int = 256, learning_rate: float = 1e-04, output_dir: str = None):
    # Load the buggy model
    model = load_buggy_models(program_name, job_id)[0]
    train_dataset = load_dataset(model_name, max_len, split_name="train")
    val_dataset = load_dataset(model_name, max_len, split_name="val")

    X_train, Y_train = zip(*train_dataset)
    X_val, Y_val = zip(*val_dataset)

    # Train the model and get metrics
    trainer = Trainer(
        model=model,
        params=model.params,
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
    )

    trainer.train()
    trainer.save_metrics()
    trainer.save_model()


@click.command()
@click.option("--model_name", type=str, help="The name of the model architecture to use")
@click.option("--program_name", type=str, help="The name of the program to load buggy models from")
@click.option("--job_id", type=int, help="The job ID for loading the specific buggy model")
def run_test(model_name, program_name, job_id):
    print(f"Training mutated model {program_name} (job {job_id}) with architecture {model_name}...")
    output_dir = f"saved_data/{program_name}/job_{job_id}/"
    train_mutated_model(
        model_name=model_name,
        program_name=program_name,
        job_id=job_id,
        output_dir=output_dir
    )

if __name__ == "__main__":
    run_test()