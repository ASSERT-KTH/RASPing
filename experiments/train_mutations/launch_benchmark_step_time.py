"""
Launch benchmark_step_time.py on the cluster for one representative job per program.

After all jobs complete, merge results into a single gbpr_timing.json:
    python -c "
    import json
    from pathlib import Path
    results = {}
    for f in Path('benchmark_results').glob('*.json'):
        d = json.load(open(f))
        results[d['program_name']] = {
            'seconds_per_epoch': d['seconds_per_epoch'],
            'seconds_per_val_step': d['seconds_per_val_step'],
        }
    json.dump(results, open('gbpr_timing.json', 'w'), indent=2)
    print(json.dumps(results, indent=2))
    "
"""
import os
import subprocess
import submitit

from pathlib import Path


# One representative job_id per program (first job in saved_data)
REPRESENTATIVE_JOBS = {
    "hist": "0532b1914bc2417486ac590b6672afb5",
    "most_freq": "010d1cd204b84af397f31e73d4505e7a",
    "reverse": "0079c3471a1b4dd38da7364d04bf48dc",
    "shuffle_dyck": "01d3a605118c41aca3b6bac7f06b86d0",
    "shuffle_dyck2": "011feaa061f24316995aa1d14301d847",
    "sort": "0152617485ba4c4f9a2c74f0f90a3c63",
}


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


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("benchmark_results", exist_ok=True)

    executor = get_executor()
    jobs = []

    for program_name, job_id in REPRESENTATIVE_JOBS.items():
        output_path = str(
            Path(__file__).parent / f"benchmark_results/{program_name}.json"
        )
        if Path(output_path).exists():
            print(f"Skipping {program_name} (already exists)")
            continue

        job = executor.submit(
            run_benchmark_in_container,
            program_name=program_name,
            job_id=job_id,
            output_path=output_path,
        )
        jobs.append((program_name, job))
        print(f"Submitted benchmark for {program_name} (job_id={job_id})")

    print(f"\nSubmitted {len(jobs)} benchmark jobs.")
    print("After completion, merge results with:")
    print(
        "  python -c \""
        "import json; from pathlib import Path; results = {}; "
        "[results.update({d['program_name']: {'seconds_per_epoch': d['seconds_per_epoch'], 'seconds_per_val_step': d['seconds_per_val_step']}}) "
        "for f in Path('benchmark_results').glob('*.json') for d in [json.load(open(f))]]; "
        "json.dump(results, open('gbpr_timing.json', 'w'), indent=2); print(json.dumps(results, indent=2))"
        "\""
    )


if __name__ == "__main__":
    main()
