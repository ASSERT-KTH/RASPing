import os
import submitit
import subprocess
import pandas as pd
import json
from pathlib import Path


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="test_logs")
    executor.update_parameters(
        name="RASPING-TEST",
        nodes=1,
        timeout_min=60,  # Testing should be much faster than training
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_in_container(
    program_name: str,
    job_id: str,
    max_len: int = 10,
    output_dir: str = None,
):
    """Wrapper to run test_trained_mutations.py inside the apptainer container"""
    # Get path to container.sif relative to repository root
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"

    # Get full path to test_trained_mutations.py
    test_mutations_path = Path(__file__).parent / "test_trained_mutations.py"

    cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        str(test_mutations_path),
        "--program_name",
        program_name,
        "--job_id",
        job_id,
        "--max_len",
        str(max_len),
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
        print(f"Error testing model for {program_name} (job {job_id}):")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def main():
    # Create output directory if it doesn't exist
    os.makedirs("test_logs", exist_ok=True)

    # Initialize the executor
    executor = get_executor()

    # Create a list of jobs to run
    jobs = []

    # Load the dataframe with mutation information
    mutation_path = (
        Path(__file__).parent / "../mutation/results/aggregated_mutations.json"
    )
    df = pd.read_json(mutation_path)

    for _, row in df.iterrows():
        # We only test models that were buggy and then trained
        if row["execution_result"].get("status") != "BUGGY_MODEL":
            continue

        program_name = row["program_name"]
        job_id = row["job_id"]

        # Check if there's a trained model for this configuration
        output_dir = f"saved_data/{program_name}/job_{job_id}/"
        model_path = Path(__file__).parent / output_dir / "model.npy"

        if not (model_path.exists()):
            print(f"Skipping {program_name} job {job_id}: no trained model found")
            continue

        # Create the job using the container wrapper
        job = executor.submit(
            run_in_container,
            program_name=program_name,
            job_id=job_id,
            max_len=10,
            output_dir=output_dir,
        )
        jobs.append(job)

    print(f"Submitted {len(jobs)} test jobs")

    # Wait for all jobs to complete and aggregate results
    results = []
    for job in jobs:
        job.wait()

    # Aggregate all test results
    all_results = []
    base_path = Path(__file__).parent / "saved_data"
    for program_dir in base_path.glob("*"):
        if not program_dir.is_dir():
            continue
        for job_dir in program_dir.glob("job_*"):
            result_file = job_dir / "test_results.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    all_results.append(json.load(f))

    # Save aggregated results
    with open(Path(__file__).parent / "test_results_aggregated.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
