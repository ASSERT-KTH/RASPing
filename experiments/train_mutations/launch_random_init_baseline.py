"""Initialization-control baseline: repair a randomly-initialized, ground-truth
(bug-free) compiled model exactly the same way buggy mutations are repaired.

Reviewer request: "Add initialization controls. Compare buggy-program
initialization against the same architecture with random initialization ...
This is necessary to establish that the method benefits from repairing the
buggy representation rather than simply learning the intended function from
labeled examples."

Matches launch_train_mutations.py's actual production hyperparameters
(n_epochs=10000, batch_size=256, lr=1e-4; early_stopping_patience left at
train_mutations.py's CLI default of 50) so this is a like-for-like
comparison against the existing buggy-init repair results in
saved_data/{program}/cross_entropy_loss/job_*/. The "unrelated compiled
program" initialization control from the same review comment was checked
and ruled out: RASP-to-transformer compilation produces a different
architecture per program (parameter counts differ by up to ~220x across
hist/reverse/sort/most-freq), so there is no way to transplant weights
between them.

Usage:
    python launch_random_init_baseline.py
    python launch_random_init_baseline.py --programs hist,sort --seeds 0,1,2
"""
import os
import subprocess
from pathlib import Path

import submitit


PROGRAMS = ["hist", "most_freq", "reverse", "sort"]
N_SEEDS = 10
N_EPOCHS = 10000  # matches launch_train_mutations.py's actual production runs
BATCH_SIZE = 256
LEARNING_RATE = 1e-04
LOSS_FN_NAME = "cross_entropy_loss"


def get_executor(timeout_min: int, array_parallelism: int) -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="RASPING-RANDOM-INIT",
        nodes=1,
        timeout_min=timeout_min,
        slurm_array_parallelism=array_parallelism,
        slurm_additional_parameters={
            "reservation": "1g.10gb",
        },
    )
    return executor


def run_train_and_test(program_name: str, seed: int, output_dir: str, init_scheme: str = "xavier"):
    """Run train_mutations.py --random-init then test_trained_mutations.py --random-init."""
    repo_root = Path(__file__).parent.parent.parent
    container_path = repo_root / "container.sif"
    train_path = Path(__file__).parent / "train_mutations.py"
    test_path = Path(__file__).parent / "test_trained_mutations.py"

    job_id = f"seed{seed}"
    full_output_dir = Path(__file__).parent / output_dir

    train_cmd = [
        "apptainer", "exec", "--nv", str(container_path),
        "python", str(train_path),
        "--program_name", program_name,
        "--job_id", job_id,
        "--max_len", "10",
        "--n_epochs", str(N_EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", str(LEARNING_RATE),
        "--loss_fn_name", LOSS_FN_NAME,
        "--seed", str(seed),
        "--no-store-trajectory",
        "--random-init",
        "--init-scheme", init_scheme,
        "--output_dir", str(full_output_dir),
    ]

    test_cmd = [
        "apptainer", "exec", "--nv", str(container_path),
        "python", str(test_path),
        "--program_name", program_name,
        "--job_id", job_id,
        "--max_len", "10",
        "--loss_fn_name", LOSS_FN_NAME,
        "--random-init",
        "--output_dir", str(full_output_dir),
    ]

    model_path = full_output_dir / "model.npy"
    if model_path.exists():
        print(f"Skipping training for {program_name} seed={seed}: model.npy already exists "
              f"(resuming straight to test - avoids redoing hours of training if a prior "
              f"attempt died between train and test)")
    else:
        try:
            subprocess.run(train_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error training {program_name} seed={seed}:")
            print(f"stdout: {e.stdout[-2000:]}")
            print(f"stderr: {e.stderr[-2000:]}")
            raise

    try:
        subprocess.run(test_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error testing {program_name} seed={seed}:")
        print(f"stdout: {e.stdout[-2000:]}")
        print(f"stderr: {e.stderr[-2000:]}")
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", type=str, default=None,
                         help=f"Comma-separated subset of {PROGRAMS} (default: all).")
    parser.add_argument("--seeds", type=str, default=None,
                         help=f"Comma-separated seeds (default: 0..{N_SEEDS - 1}).")
    parser.add_argument("--timeout-min", type=int, default=4200,
                         help="Per-job Slurm timeout (70h). Worst case at full 10000 epochs "
                              "is ~15.4h (most_freq); this leaves large headroom for shared-"
                              "reservation contention slowdowns while staying under the "
                              "cluster's 72h MaxTime. A timeout here loses the whole job's "
                              "work (no mid-training checkpointing), so err generous.")
    parser.add_argument("--init-scheme", type=str, default="xavier",
                         choices=["lecun", "xavier", "kaiming", "unit-normal"],
                         help="Weight init for the random-init control. Default 'xavier' "
                              "(fan-in scaled). 'unit-normal' is the historical N(0,1), which "
                              "is not fan-in scaled and starts these models numerically broken.")
    parser.add_argument("--output-root", type=str, default="saved_data_random_init",
                         help="Directory under experiments/train_mutations/ to write results to.")
    parser.add_argument("--array-parallelism", type=int, default=40,
                         help="Max concurrent tasks on the shared 1g.10gb reservation.")
    args = parser.parse_args()

    programs = args.programs.split(",") if args.programs else PROGRAMS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(N_SEEDS))

    os.makedirs("logs", exist_ok=True)
    executor = get_executor(timeout_min=args.timeout_min, array_parallelism=args.array_parallelism)

    jobs = []
    with executor.batch():
        for program_name in programs:
            for seed in seeds:
                output_dir = f"{args.output_root}/{program_name}/seed_{seed}/{LOSS_FN_NAME}/job_seed{seed}"
                if (Path(__file__).parent / output_dir / "test_results.json").exists():
                    print(f"Skipping {program_name} seed={seed}: already completed")
                    continue
                job = executor.submit(run_train_and_test, program_name, seed, output_dir, args.init_scheme)
                jobs.append((program_name, seed, job))

    print(f"Submitted {len(jobs)} jobs ({len(programs)} programs x {len(seeds)} seeds), "
          f"init-scheme={args.init_scheme}.")
    print(f"Results land in {args.output_root}/{{program}}/seed_{{seed}}/{LOSS_FN_NAME}/job_seed{{seed}}/")


if __name__ == "__main__":
    main()
