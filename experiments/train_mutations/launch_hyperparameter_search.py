import os
import submitit
import subprocess
import pandas as pd
import itertools
from pathlib import Path
from typing import List, Dict, Any


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="RASPING-HYPERPARAM",
        nodes=1,
        timeout_min=12 * 60,
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_in_container(
    program_name: str,
    job_id: str,
    max_len: int = 10,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-04,
    temperature: float = 1.0,
    entropy_coef: float = 0.01,
    baseline: str = "none",
    output_dir: str = None,
    reward_fn_name: str = "exact_match",
    from_pretrained: str = None,
):
    """Wrapper to run train_reinforce.py inside the apptainer container"""
    # Get path to container.sif relative to repository root
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"

    # Get full path to train_reinforce.py
    train_reinforce_path = Path(__file__).parent / "train_reinforce.py"

    cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        str(train_reinforce_path),
        "--program_name",
        program_name,
        "--job_id",
        job_id,
        "--max_len",
        str(max_len),
        "--n_epochs",
        str(n_epochs),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--temperature",
        str(temperature),
        "--entropy_coef",
        str(entropy_coef),
        "--baseline",
        baseline,
        "--reward_fn_name",
        reward_fn_name,
    ]

    # Add output directory if specified, using full path
    if output_dir:
        full_output_dir = Path(__file__).parent / output_dir
        cmd.extend(["--output_dir", str(full_output_dir)])
    
    # Add from_pretrained if specified
    if from_pretrained:
        cmd.extend(["--from_pretrained", from_pretrained])

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running container for {program_name} (job {job_id}):")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def generate_parameter_combinations() -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters to search over"""
    # Define the parameter ranges
    param_ranges = {
        'batch_size': [16, 32, 64],
        'learning_rate': [1e-4, 5e-5, 1e-5],
        'temperature': [0.5, 1.0, 2.0],
        'entropy_coef': [0.001, 0.01, 0.1],
        'reward_fn_name': ['exact_match', 'token_match', 'binary_token', 'weighted_position'],
    }
    
    # Generate all combinations
    keys = param_ranges.keys()
    values = param_ranges.values()
    combinations = list(itertools.product(*values))
    
    # Convert to list of dictionaries
    return [dict(zip(keys, combo)) for combo in combinations]


def main():
    # Create output directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Initialize the executor
    executor = get_executor()

    # Create a list of jobs to run
    jobs = []

    # Load the dataframe
    mutation_path = (
        Path(__file__).parent / "../mutation/results/aggregated_mutations.json"
    )
    df = pd.read_json(mutation_path)

    # Generate all parameter combinations
    param_combinations = generate_parameter_combinations()
    print(f"Generated {len(param_combinations)} parameter combinations")

    for _, row in df.iterrows():
        # We only train buggy models
        if row["execution_result"].get("status") != "BUGGY_MODEL":
            continue

        program_name = row["program_name"]
        job_id = row["job_id"]

        # Submit a job for each parameter combination
        for params in param_combinations:
            # Create a unique identifier for this parameter combination
            param_id = f"b{params['batch_size']}_lr{params['learning_rate']}_t{params['temperature']}_e{params['entropy_coef']}_{params['reward_fn_name']}"
            output_dir = Path(__file__).parent / f"saved_data/{program_name}/hyperparam_search/{param_id}/job_{job_id}"
            
            if output_dir.exists():
                print(f"Skipping {program_name} (job {job_id}) with {param_id} because it already exists")
                continue

            # Create the job using the container wrapper
            job = executor.submit(
                run_in_container,
                program_name=program_name,
                job_id=job_id,
                max_len=10,
                n_epochs=100,
                **params,
                baseline="none",
                output_dir=f"saved_data/{program_name}/hyperparam_search/{param_id}/job_{job_id}/",
            )
            jobs.append(job)

        # Only submit jobs for one program
        break

    print(f"Submitted {len(jobs)} jobs")


if __name__ == "__main__":
    main() 