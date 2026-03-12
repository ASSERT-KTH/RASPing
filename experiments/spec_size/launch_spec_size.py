import os
import submitit
import subprocess
import pandas as pd
from pathlib import Path


N_VALUES = [100, 1000, 5000, 10000, 25000]
MAX_N_BY_PROGRAM = {"shuffle_dyck": 1635}
SEEDS = [42]          # extend to [42, 123, 456] for multi-seed error bars later
N_STEPS = 500_000     # fixed gradient-step budget for all (mutation, N) pairs


def get_executor() -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="RASPING-SPEC-SIZE",
        nodes=1,
        timeout_min=12 * 60,
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_train_and_test(
    program_name: str,
    job_id: str,
    n_train_samples: int,
    seed: int,
    n_val_samples: int = None,
    n_steps: int = N_STEPS,
    max_len: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-04,
    loss_fn_name: str = "cross_entropy_loss",
    output_dir: str = None,
):
    """Run train_mutations.py then test_trained_mutations.py inside the apptainer container."""
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"

    train_mutations_path = repo_root / "experiments/train_mutations/train_mutations.py"
    test_mutations_path = repo_root / "experiments/train_mutations/test_trained_mutations.py"

    # Resolve full output_dir path
    if output_dir:
        full_output_dir = Path(__file__).parent / output_dir
    else:
        full_output_dir = (
            Path(__file__).parent
            / f"saved_data/{program_name}/n_{n_train_samples}/seed_{seed}/{loss_fn_name}/job_{job_id}"
        )

    train_cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        str(train_mutations_path),
        "--program_name", program_name,
        "--job_id", job_id,
        "--max_len", str(max_len),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--loss_fn_name", loss_fn_name,
        "--n_train_samples", str(n_train_samples),
        "--seed", str(seed),
        "--n_steps", str(n_steps),
        *(["--n_val_samples", str(n_val_samples)] if n_val_samples is not None else []),
        "--no-store-trajectory",
        "--output_dir", str(full_output_dir),
    ]

    test_cmd = [
        "apptainer",
        "exec",
        "--nv",
        str(container_path),
        "python",
        str(test_mutations_path),
        "--program_name", program_name,
        "--job_id", job_id,
        "--max_len", str(max_len),
        "--loss_fn_name", loss_fn_name,
        "--output_dir", str(full_output_dir),
    ]

    try:
        subprocess.run(train_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error training {program_name} (job {job_id}) n={n_train_samples} seed={seed}:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

    try:
        subprocess.run(test_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error testing {program_name} (job {job_id}) n={n_train_samples} seed={seed}:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def main():
    os.makedirs("logs", exist_ok=True)

    executor = get_executor()

    jobs = []

    mutation_path = Path(__file__).parent / "../mutation/results/aggregated_mutations.json"
    df = pd.read_json(mutation_path)

    loss_fn_name = "cross_entropy_loss"

    for _, row in df.iterrows():
        if row["execution_result"].get("status") != "BUGGY_MODEL":
            continue

        program_name = row["program_name"]
        job_id = row["job_id"]

        for n in N_VALUES:
            if n > MAX_N_BY_PROGRAM.get(program_name, float("inf")):
                continue
            for seed in SEEDS:
                output_dir = (
                    f"saved_data/{program_name}/n_{n}/seed_{seed}/{loss_fn_name}/job_{job_id}"
                )
                # Skip if test_results.json already exists
                results_path = Path(__file__).parent / output_dir / "test_results.json"
                if results_path.exists():
                    print(
                        f"Skipping {program_name} (job {job_id}) n={n} seed={seed}: already completed"
                    )
                    continue

                n_val_samples = n // 8
                job = executor.submit(
                    run_train_and_test,
                    program_name=program_name,
                    job_id=job_id,
                    n_train_samples=n,
                    seed=seed,
                    n_val_samples=n_val_samples,
                    n_steps=N_STEPS,
                    max_len=10,
                    batch_size=256,
                    learning_rate=1e-04,
                    loss_fn_name=loss_fn_name,
                    output_dir=output_dir,
                )
                jobs.append(job)

    print(f"Submitted {len(jobs)} jobs")


if __name__ == "__main__":
    main()
