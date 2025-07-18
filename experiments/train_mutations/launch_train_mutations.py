import os
import submitit
import subprocess
import pandas as pd

from pathlib import Path


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="RASPING-MUTATIONS",
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
    n_epochs: int = 50000,
    batch_size: int = 256,
    learning_rate: float = 1e-04,
    output_dir: str = None,
):
    """Wrapper to run train_mutations.py inside the apptainer container"""
    # Get path to container.sif relative to repository root
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"

    # Get full path to train_mutations.py
    train_mutations_path = Path(__file__).parent / "train_mutations.py"

    cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        str(train_mutations_path),
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
    ]

    # Add output directory if specified, using full path
    if output_dir:
        full_output_dir = Path(__file__).parent / output_dir
        cmd.extend(["--output_dir", str(full_output_dir)])

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

    for _, row in df.iterrows():
        # We only train buggy models
        if row["execution_result"].get("status") != "BUGGY_MODEL":
            continue

        # Create the job using the container wrapper
        program_name = row["program_name"]
        job_id = row["job_id"]
        job = executor.submit(
            run_in_container,
            program_name=program_name,
            job_id=job_id,
            max_len=10,
            n_epochs=10000,
            batch_size=256,
            learning_rate=1e-04,
            output_dir=f"saved_data/{program_name}/job_{job_id}/",
        )
        jobs.append(job)

    print(f"Submitted {len(jobs)} jobs")


if __name__ == "__main__":
    # No need to import train_mutated_model since we're running it through apptainer
    main()
