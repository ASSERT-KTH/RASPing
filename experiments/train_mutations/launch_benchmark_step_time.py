"""
Launch benchmark_step_time.py on the cluster for 10 jobs per base program (60 total).
Timing is averaged across the 10 jobs per program to get a robust estimate.

After all jobs complete, merge and average results into gbpr_timing.json:
    python experiments/train_mutations/launch_benchmark_step_time.py --merge
"""
import os
import json
import argparse
import subprocess
import submitit
import numpy as np

from pathlib import Path


N_JOBS_PER_PROGRAM = 10
SAVED_DATA_DIR = Path(__file__).parent / "saved_data"
BENCHMARK_RESULTS_DIR = Path(__file__).parent / "benchmark_results"
LOSS_FN = "cross_entropy_loss"


def get_job_ids_for_program(program_name: str, n: int) -> list[str]:
    """Sample n job_ids evenly from saved_data for a given program."""
    loss_dir = SAVED_DATA_DIR / program_name / LOSS_FN
    all_jobs = sorted(loss_dir.iterdir())
    if len(all_jobs) <= n:
        selected = all_jobs
    else:
        indices = np.linspace(0, len(all_jobs) - 1, n, dtype=int)
        selected = [all_jobs[i] for i in indices]
    return [j.name.replace("job_", "") for j in selected]


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="RASPING-BENCHMARK",
        nodes=1,
        timeout_min=60,
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_benchmark_in_container(
    program_name: str,
    job_id: str,
    output_path: str,
    warmup_epochs: int = 50,
    measure_epochs: int = 200,
):
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"
    benchmark_path = Path(__file__).parent / "benchmark_step_time.py"

    cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        str(benchmark_path),
        "--program_name", program_name,
        "--job_id", job_id,
        "--warmup_epochs", str(warmup_epochs),
        "--measure_epochs", str(measure_epochs),
        "--output_path", output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error for {program_name} ({job_id}):")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def merge_results(output_path: str = "gbpr_timing.json"):
    """Average per-job benchmark results into one timing file per program."""
    per_program: dict[str, list[dict]] = {}
    for f in sorted(BENCHMARK_RESULTS_DIR.glob("*.json")):
        d = json.load(open(f))
        prog = d["program_name"]
        per_program.setdefault(prog, []).append(d)

    results = {}
    for prog, records in sorted(per_program.items()):
        spe = np.mean([r["seconds_per_epoch"] for r in records])
        spvs = np.mean([r["seconds_per_val_step"] for r in records])
        results[prog] = {
            "n_jobs": len(records),
            "seconds_per_epoch": float(spe),
            "seconds_per_val_step": float(spvs),
        }
        print(f"{prog}: {len(records)} jobs, {spe:.3f} s/epoch, {spvs:.3f} s/val_step")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge completed benchmark results into gbpr_timing.json instead of launching jobs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gbpr_timing.json",
        help="Output path for merged timing file (used with --merge)",
    )
    args = parser.parse_args()

    if args.merge:
        merge_results(args.output)
        return

    os.makedirs("logs", exist_ok=True)
    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    executor = get_executor()
    submitted = []

    programs = [d.name for d in sorted(SAVED_DATA_DIR.iterdir()) if d.is_dir()]
    for program_name in programs:
        job_ids = get_job_ids_for_program(program_name, N_JOBS_PER_PROGRAM)
        for job_id in job_ids:
            output_path = str(BENCHMARK_RESULTS_DIR / f"{program_name}_{job_id}.json")
            if Path(output_path).exists():
                print(f"Skipping {program_name}/{job_id} (already exists)")
                continue

            job = executor.submit(
                run_benchmark_in_container,
                program_name=program_name,
                job_id=job_id,
                output_path=output_path,
            )
            submitted.append((program_name, job_id, job))
            print(f"Submitted {program_name}/{job_id}")

    print(f"\nSubmitted {len(submitted)} benchmark jobs ({N_JOBS_PER_PROGRAM} per program).")
    print("After completion, run:")
    print("  python experiments/train_mutations/launch_benchmark_step_time.py --merge")


if __name__ == "__main__":
    main()
