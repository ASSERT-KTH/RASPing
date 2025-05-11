import os
import submitit
import subprocess
from pathlib import Path
import argparse


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs_plotting_jobs")
    executor.update_parameters(
        name="RASPING-PLOT-TRAJECTORIES",
        nodes=1,
        timeout_min=10,  # 10 minutes
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_plot_in_container(trajectory_file: str, output_dir: str):
    """Run the plot_trajectories.py plot_file command inside the apptainer container."""
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"
    plot_trajectories_path = Path(__file__).parent / "plot_trajectories.py"

    cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        str(plot_trajectories_path),
        "plot_file",
        trajectory_file,
        output_dir,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running plotter for {trajectory_file}:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Launch plotting jobs for all trajectory.pkl files in a directory.")
    parser.add_argument("--data-dir", required=True, help="Directory to search for trajectory.pkl files")
    parser.add_argument("--output-dir", required=True, help="Directory to save plot outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = args.output_dir

    # Find all trajectory.pkl files recursively
    trajectory_files = list(data_dir.rglob("trajectory.pkl"))
    if not trajectory_files:
        print(f"No trajectory.pkl files found in {data_dir} or its subdirectories.")
        return

    print(f"Found {len(trajectory_files)} trajectory.pkl files. Submitting jobs...")

    executor = get_executor()
    jobs = []
    for trajectory_file in trajectory_files:
        job = executor.submit(run_plot_in_container, str(trajectory_file), output_dir)
        jobs.append(job)
        print(f"Submitted job for {trajectory_file}")

    print(f"Submitted {len(jobs)} plotting jobs.")


if __name__ == "__main__":
    main() 