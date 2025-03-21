import os
import submitit
import subprocess
import pandas as pd

from pathlib import Path


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="RASPING-REINFORCE",
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

    # List of reward functions to test
    reward_functions = [
        "exact_match",
        "token_match", 
        "binary_token",
        "weighted_position",
    ]

    for _, row in df.iterrows():
        # We only train buggy models
        if row["execution_result"].get("status") != "BUGGY_MODEL":
            continue

        program_name = row["program_name"]
        job_id = row["job_id"]

        # Submit a job for each reward function
        for reward_fn_name in reward_functions:
            output_dir = Path(__file__).parent / f"saved_data/{program_name}/reinforce_{reward_fn_name}/job_{job_id}"
            if output_dir.exists():
                print(f"Skipping {program_name} (job {job_id}) with {reward_fn_name} because it already exists")
                continue

            # Create the job using the container wrapper
            job = executor.submit(
                run_in_container,
                program_name=program_name,
                job_id=job_id,
                max_len=10,
                n_epochs=1000,
                batch_size=32,
                learning_rate=1e-04,
                temperature=1.0,
                entropy_coef=0.01,
                baseline="none",
                output_dir=f"saved_data/{program_name}/reinforce_{reward_fn_name}/job_{job_id}/",
                reward_fn_name=reward_fn_name,
            )
            jobs.append(job)

    print(f"Submitted {len(jobs)} jobs")


if __name__ == "__main__":
    main() 