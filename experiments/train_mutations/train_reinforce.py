import sys
import os
import click

from pathlib import Path

# TODO: this is a hack, change this for editable install
module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.functions import load_dataset, encodeAndPadData
from src.reinforcement_trainer import REINFORCETrainer
from src.rewards import exact_match_reward, token_match_reward, binary_token_reward, weighted_position_reward
from experiments.mutation.load_mutations import load_buggy_models

# Define reward functions
REWARD_FUNCTIONS = {
    "exact_match": exact_match_reward,
    "token_match": token_match_reward, 
    "binary_token": binary_token_reward,
    "weighted_position": weighted_position_reward,
}


def train_mutated_model_with_reinforce(
    program_name: str,
    job_id: str,
    max_len: int = 10,
    n_epochs: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-04,
    temperature: float = 1.0,
    entropy_coef: float = 0.01,
    baseline: str = "none",
    output_dir: str = None,
    reward_fn_name: str = "exact_match",
    from_pretrained: str = None,
):
    # Validate reward function
    if reward_fn_name not in REWARD_FUNCTIONS:
        raise ValueError(f"Reward function {reward_fn_name} not found")
    reward_fn = REWARD_FUNCTIONS[reward_fn_name]

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

    # Train the model with REINFORCE
    trainer = REINFORCETrainer(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        reward_fn=reward_fn,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        temperature=temperature,
        entropy_coef=entropy_coef,
        baseline=baseline,
        output_dir=output_dir,
        use_wandb=True,
        wandb_project="dpr-mutation-reinforce",
        wandb_name=f"{program_name}_{job_id}_{reward_fn_name}",
        from_pretrained=from_pretrained,
    )

    # Train model and get metrics
    results = trainer.train()
    
    return results


@click.command()
@click.option(
    "--program_name", type=str, help="The name of the program to load buggy models from"
)
@click.option(
    "--job_id", type=str, help="The job ID for loading the specific buggy model"
)
@click.option("--max_len", type=int, default=10, help="Maximum sequence length")
@click.option("--n_epochs", type=int, default=1000, help="Number of training epochs")
@click.option("--batch_size", type=int, default=32, help="Training batch size")
@click.option("--learning_rate", type=float, default=1e-04, help="Learning rate")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature")
@click.option("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
@click.option(
    "--baseline",
    type=click.Choice(["none", "mean", "value"]),
    default="none",
    help="Baseline type for reward normalization",
)
@click.option("--output_dir", type=str, help="Directory to save training outputs")
@click.option(
    "--reward_fn_name",
    type=click.Choice(list(REWARD_FUNCTIONS.keys())),
    default="exact_match",
    help="Reward function to use for training",
)
@click.option(
    "--from_pretrained", 
    type=str, 
    default=None, 
    help="Path to load pretrained model weights from"
)
def run_reinforce(
    program_name,
    job_id,
    max_len,
    n_epochs,
    batch_size,
    learning_rate,
    temperature,
    entropy_coef,
    baseline,
    output_dir,
    reward_fn_name,
    from_pretrained,
):
    print(f"Training mutated model {program_name} (job {job_id}) with REINFORCE using {reward_fn_name} reward...")
    if not output_dir:
        output_dir = f"saved_data/{program_name}/reinforce_{reward_fn_name}/job_{job_id}/"
    
    train_mutated_model_with_reinforce(
        program_name=program_name,
        job_id=job_id,
        max_len=max_len,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        entropy_coef=entropy_coef,
        baseline=baseline,
        output_dir=output_dir,
        reward_fn_name=reward_fn_name,
        from_pretrained=from_pretrained,
    )


if __name__ == "__main__":
    run_reinforce() 