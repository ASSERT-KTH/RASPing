import os
import submitit
import subprocess
import pandas as pd

from pathlib import Path
from typing import Tuple, List

# Program names and their corresponding job IDs from test_load_mutations.py
PROGRAM_CONFIGS: List[Tuple[str, str]] = [
    ("sort", "f35ba7e838874bba8335cd9ca5db2aa7"),
    ("reverse", "9db50f10314547858e52a5aff4bc2be4"),
    ("hist", "45728e11fb1043829d4057c016b549b9"),
    ("most_freq", "14ac16fe5f49412aa1ee30461b5769a0"),
    ("shuffle_dyck", "8940b2d2299f45c08b474d151c6d760b"),
    ("shuffle_dyck2", "739a764b78784fc3b4b6f80006eac399"),
]


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="RASPING-MUTATIONS",
        nodes=1,
        timeout_min=12 * 60,
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
        mem_gb=10,
        cpus_per_task=1,
        gpus_per_node=1,
    )
    return executor


def run_in_container(
    model_name: str,
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

    cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        "train_mutations.py",
        "--model_name",
        model_name,
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

    # Add output directory if specified
    if output_dir:
        cmd.extend(["--output_dir", output_dir])

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,  # Run in same directory as this script
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

    for program_name, job_id in PROGRAM_CONFIGS:
        # The model_name is the same as program_name for our datasets
        model_name = program_name

        # Create the job using the container wrapper
        job = executor.submit(
            run_in_container,
            model_name=model_name,
            program_name=program_name,
            job_id=job_id,
            max_len=10,
            n_epochs=50000,
            batch_size=256,
            learning_rate=1e-04,
            output_dir=f"saved_data/{program_name}/job_{job_id}/",
        )
        jobs.append(job)

    print(f"Submitted {len(jobs)} jobs")


if __name__ == "__main__":
    # No need to import train_mutated_model since we're running it through apptainer
    main()
