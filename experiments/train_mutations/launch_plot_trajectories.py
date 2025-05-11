#!/usr/bin/env python
import submitit
from pathlib import Path
import click

from plot_trajectories import plot_single_trajectory


def get_plot_executor(log_folder: str = "logs_plotting_jobs") -> submitit.AutoExecutor:
    """Configures and returns a submitit.AutoExecutor for plotting jobs."""
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        name="RASPING-PLOT-TRAJECTORIES",
        nodes=1,
        timeout_min=10,
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_plotting_job(trajectory_file_path_str: str, output_dir_str: str):
    """
    Wrapper function to be executed by submitit for each plotting job.
    Calls plot_single_trajectory from plot_trajectories.py.
    """
    print(f"Plotting job started for: {trajectory_file_path_str}")
    print(f"Output directory for this job: {output_dir_str}")
    
    # plot_single_trajectory handles its own logging and returns True/False
    # The job will be marked as 'DONE' by SLURM/submitit regardless of True/False,
    # as long as this wrapper doesn't raise an unhandled exception.
    # Errors within plot_single_trajectory are logged by that function.
    success = plot_single_trajectory(trajectory_file_path_str, output_dir_str)
    
    if success:
        print(f"Plotting job completed successfully for: {trajectory_file_path_str}")
    else:
        print(f"Plotting job finished with errors for: {trajectory_file_path_str}. Check logs from plot_single_trajectory.")


@click.command()
@click.option(
    "--data-dir",
    required=True,
    help="Directory containing trajectory.pkl files (searched recursively).",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    "--output-dir",
    required=True,
    help="Base directory to save plot outputs. Subdirectories will be created by plot_single_trajectory.",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    "--program",
    help="Filter by program name (optional).",
    default=None,
    type=str
)
@click.option(
    "--loss-function",
    help="Filter by loss function name (optional).",
    default=None,
    type=str
)
@click.option(
    "--job-id",
    help="Filter by job ID (from the training job, part of the path, optional).",
    default=None,
    type=str
)
@click.option(
    "--log-folder",
    default="logs_plotting_jobs", # Default submitit log folder name
    help="Folder to store submitit logs for the plotting jobs.",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True)
)
def launch_plots(data_dir: str, output_dir: str, program: str | None, loss_function: str | None, job_id: str | None, log_folder: str):
    """
    Launches multiple trajectory plotting jobs using submitit,
    one for each trajectory.pkl file found in data_dir, with optional filters.
    """
    # Ensure base output directory exists (log_folder is created in get_plot_executor)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    executor = get_plot_executor(log_folder=str(Path(log_folder).resolve()))
    submitted_jobs = []

    print(f"Scanning for trajectory.pkl files in: {data_dir}")
    
    # Convert data_dir string to Path object for rglob
    data_dir_path = Path(data_dir)
    try:
        trajectory_files = list(data_dir_path.rglob("trajectory.pkl"))
    except Exception as e:
        print(f"Error scanning directory {data_dir_path}: {e}")
        return

    if not trajectory_files:
        print(f"No trajectory.pkl files found in {data_dir_path} or its subdirectories.")
        return

    print(f"Found {len(trajectory_files)} trajectory.pkl files. Applying filters and submitting jobs...")

    submitted_count = 0
    for trajectory_file_path_obj in trajectory_files:
        trajectory_file_path_str = str(trajectory_file_path_obj)

        try:
            # Path structure: .../program_name/loss_function_name/job_id/trajectory.pkl
            parts = trajectory_file_path_obj.parts
            if len(parts) < 4: # job_id, loss_function, program_name + at least one parent
                # print(f"Debug: Path parts for {trajectory_file_path_str}: {parts}")
                print(f"Warning: Path {trajectory_file_path_str} is not in the expected format (e.g., .../prog/loss/job_id/trajectory.pkl). Skipping.")
                continue
                
            file_job_id = parts[-2]
            file_loss_function = parts[-3]
            file_program_name = parts[-4]

            # Apply filters
            if program and file_program_name != program:
                continue
            if loss_function and file_loss_function != loss_function:
                continue
            if job_id and file_job_id != job_id:
                continue
            
            # The output_dir for plot_single_trajectory is the main --output-dir.
            # plot_single_trajectory will create its own subdirectories for program_name/loss_function etc.
            # Pass resolved absolute paths to the job.
            job = executor.submit(
                run_plotting_job,
                trajectory_file_path_str=str(trajectory_file_path_obj.resolve()),
                output_dir_str=str(Path(output_dir).resolve())
            )
            submitted_jobs.append(job)
            submitted_count += 1
            print(f"Submitted plotting job for: {trajectory_file_path_str} (Job ID: {job.job_id})")

        except IndexError:
            print(f"Warning: Could not parse path structure for {trajectory_file_path_str} (parts: {parts}). Skipping.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while preparing job for {trajectory_file_path_str}: {e}. Skipping.")
            import traceback
            traceback.print_exc()


    if submitted_count > 0:
        print(f"Successfully submitted {submitted_count} plotting jobs.")
        print(f"Monitor submitit logs in: {Path(log_folder).resolve()}")
        print("Submitted job details:")
        for job in submitted_jobs:
            # Accessing kwargs might be tricky if job submission is very async, but usually fine.
            try:
                traj_file_arg = job.kwargs.get('trajectory_file_path_str', 'N/A') if hasattr(job, 'kwargs') else 'N/A'
                print(f"  SLURM Job ID: {job.job_id}, Trajectory File: {traj_file_arg}")
            except Exception:
                 print(f"  SLURM Job ID: {job.job_id} (could not retrieve trajectory file from job object)")
    elif len(trajectory_files) > 0: # Files found but none were submitted
        print(f"Found {len(trajectory_files)} trajectory files, but no plotting jobs were submitted. Check filter criteria or logs.")
    else: # No files were found initially (already handled, but for clarity)
        print("No trajectory files found, so no jobs were submitted.")


if __name__ == "__main__":
    # This script assumes it's run from an environment where 'plot_trajectories.py'
    # can be imported (e.g., they are in the same directory, or PYTHONPATH is set up).
    # The 'plot_trajectories.py' script itself handles 'src' imports by modifying sys.path.
    launch_plots() 