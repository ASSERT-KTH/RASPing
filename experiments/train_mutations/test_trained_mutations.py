import sys
import os
import click
import json
import numpy as np
from pathlib import Path

# TODO: this is a hack, change this for editable install
module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.functions import load_dataset, encodeAndPadData
from experiments.mutation.load_mutations import load_buggy_models
from src.loss import (
    cross_entropy_loss,
    cross_entropy_loss_smoothed_accuracy,
    cross_entropy_loss_with_perfect_sequence,
)


LOSS_FUNCTIONS = {
    "cross_entropy_loss": cross_entropy_loss,
    "cross_entropy_loss_smoothed_accuracy": cross_entropy_loss_smoothed_accuracy,
    "cross_entropy_loss_with_perfect_sequence": cross_entropy_loss_with_perfect_sequence,
}


def test_trained_model(
    program_name: str,
    job_id: str,
    max_len: int = 10,
    output_dir: str = None,
    loss_fn_name: str = "cross_entropy_loss",
):
    # Load the trained model
    if not output_dir:
        output_dir = f"saved_data/{program_name}/{loss_fn_name}/job_{job_id}"

    output_dir = Path(output_dir)
    model_path = output_dir / "model.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}")

    # Load the base buggy model first
    model = load_buggy_models(
        max_length=max_len, program_name=program_name, job_id=job_id
    )[job_id]

    # Load the trained model parameters
    trained_params = np.load(str(model_path), allow_pickle=True).item()
    # Set the trained parameters
    model.model.params = trained_params

    # Load test dataset
    program_name_key = program_name
    if program_name == "most_freq":
        program_name_key = "most-freq"
    elif program_name == "shuffle_dyck":
        program_name_key = "shuffle_dyck1"
    data_path = f"{Path(__file__).parent.resolve()}/../../data/"
    test_dataset = load_dataset(data_path, program_name_key, split_name="test")

    # Encode the dataset
    X_test, Y_test = encodeAndPadData(
        test_dataset, model.raspFunction, model.inputs, max_len
    )

    # Evaluate the model
    accuracy = model.fastEvaluateEncoded(X_test, Y_test)

    # Save results
    results = {
        "program_name": program_name,
        "job_id": job_id,
        "test_accuracy": float(accuracy),
        "test_samples": len(X_test),
        "loss_function": loss_fn_name,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


@click.command()
@click.option("--program_name", type=str, help="The name of the program to test")
@click.option("--job_id", type=str, help="The job ID for loading the specific model")
@click.option("--max_len", type=int, default=10, help="Maximum sequence length")
@click.option(
    "--output_dir",
    type=str,
    help="Directory containing the trained model and where to save test results",
)
@click.option(
    "--loss_fn_name",
    type=click.Choice(list(LOSS_FUNCTIONS.keys())),
    default="cross_entropy_loss",
    help="Loss function used for training",
)
def run_test(
    program_name,
    job_id,
    max_len,
    output_dir,
    loss_fn_name,
):
    print(f"Testing trained model {program_name} (job {job_id}) with {loss_fn_name}...")
    results = test_trained_model(
        program_name=program_name,
        job_id=job_id,
        max_len=max_len,
        output_dir=output_dir,
        loss_fn_name=loss_fn_name,
    )
    print(f"Test results: {results}")


if __name__ == "__main__":
    run_test()
