import subprocess
from pathlib import Path

import submitit

NUM_SHARDS_PER_PROGRAM = 16
PROGRAMS = ["shuffle_dyck", "shuffle_dyck2"]


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="dyck_balanced_logs")
    executor.update_parameters(
        name="RASPING-DYCK-BALANCED",
        nodes=1,
        timeout_min=60,
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_in_container(program_name: str, shard_index: int, num_shards: int):
    """Wrapper to run analyze_dyck_balanced.py inside the apptainer container"""
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"
    script_path = Path(__file__).parent / "analyze_dyck_balanced.py"

    cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        str(script_path),
        "--program",
        program_name,
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running shard {shard_index}/{num_shards} for {program_name}:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def main():
    executor = get_executor()

    jobs = []
    with executor.batch():
        for program_name in PROGRAMS:
            for shard_index in range(NUM_SHARDS_PER_PROGRAM):
                job = executor.submit(
                    run_in_container, program_name, shard_index, NUM_SHARDS_PER_PROGRAM
                )
                jobs.append(job)

    print(f"Submitted {len(jobs)} shard jobs ({len(PROGRAMS)} programs x {NUM_SHARDS_PER_PROGRAM} shards).")
    print("Once all jobs complete, merge with:")
    print(
        "  python -c \"import pandas as pd, glob; "
        "df = pd.concat([pd.read_csv(f) for f in glob.glob('analysis_outputs/dyck_balanced_metrics_shard*-of-*.csv')]); "
        "df.to_csv('analysis_outputs/dyck_balanced_metrics.csv', index=False); print(len(df), 'rows merged')\""
    )


if __name__ == "__main__":
    main()
