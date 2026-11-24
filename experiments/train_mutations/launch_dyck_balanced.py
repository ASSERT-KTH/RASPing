import subprocess
from pathlib import Path

import submitit

NUM_SHARDS_PER_PROGRAM = 16
PROGRAMS = ["shuffle_dyck", "shuffle_dyck2"]


def get_executor(timeout_min: int = 60, array_parallelism: int = 16) -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="dyck_balanced_logs")
    executor.update_parameters(
        name="RASPING-DYCK-BALANCED",
        nodes=1,
        timeout_min=timeout_min,
        slurm_array_parallelism=array_parallelism,
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_in_container(program_name: str, shard_index: int, num_shards: int, job_timeout_sec: int):
    """Wrapper to run analyze_dyck_balanced.py inside the apptainer container"""
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"
    script_path = Path(__file__).parent / "analyze_dyck_balanced.py"
    out_path = (
        Path(__file__).parent
        / "analysis_outputs"
        / f"dyck_balanced_metrics_{program_name}_shard{shard_index}-of-{num_shards}.csv"
    )

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
        "--out",
        str(out_path),
        "--job-timeout-sec",
        str(job_timeout_sec),
    ]

    max_process_retries = 8
    for attempt in range(1, max_process_retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        print(f"Attempt {attempt}/{max_process_retries} for shard {shard_index}/{num_shards} "
              f"({program_name}) exited {result.returncode}:")
        print(f"stdout: {result.stdout[-2000:]}")
        print(f"stderr: {result.stderr[-2000:]}")
        if result.returncode != 2:
            # Not the CUDA-poisoned-process signal analyze_dyck_balanced.py uses (exit 2);
            # some other failure, don't blindly retry.
            raise RuntimeError(
                f"Shard {shard_index}/{num_shards} ({program_name}) failed with unexpected exit code "
                f"{result.returncode}"
            )
    raise RuntimeError(
        f"Shard {shard_index}/{num_shards} ({program_name}) still failing after {max_process_retries} "
        "fresh-process retries"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--program", choices=PROGRAMS, default=None,
                         help="Only submit shards for this program (default: both).")
    parser.add_argument("--shards", type=str, default=None,
                         help="Comma-separated shard indices to (re)submit (default: all).")
    parser.add_argument("--timeout-min", type=int, default=60)
    parser.add_argument("--array-parallelism", type=int, default=16,
                         help="Max concurrent tasks on the shared MIG reservation. Lower this if CUDA-init "
                              "contention causes jobs to hang (see job-timeout-sec).")
    parser.add_argument("--job-timeout-sec", type=int, default=300,
                         help="Per-model hard timeout inside each shard, passed to analyze_dyck_balanced.py.")
    args = parser.parse_args()

    programs = [args.program] if args.program else PROGRAMS
    shard_indices = (
        [int(s) for s in args.shards.split(",")] if args.shards else list(range(NUM_SHARDS_PER_PROGRAM))
    )

    executor = get_executor(timeout_min=args.timeout_min, array_parallelism=args.array_parallelism)

    jobs = []
    with executor.batch():
        for program_name in programs:
            for shard_index in shard_indices:
                job = executor.submit(
                    run_in_container, program_name, shard_index, NUM_SHARDS_PER_PROGRAM, args.job_timeout_sec
                )
                jobs.append(job)

    print(f"Submitted {len(jobs)} shard jobs ({len(programs)} programs x {len(shard_indices)} shards).")
    print("Once all jobs complete, merge with:")
    print(
        "  python -c \"import pandas as pd, glob; "
        "df = pd.concat([pd.read_csv(f) for f in glob.glob('analysis_outputs/dyck_balanced_metrics_*_shard*-of-*.csv')]); "
        "df.to_csv('analysis_outputs/dyck_balanced_metrics.csv', index=False); print(len(df), 'rows merged')\""
    )


if __name__ == "__main__":
    main()
