"""Paired initialization control: randomize each mutant's OWN architecture.

The earlier random-init baseline (launch_random_init_baseline.py) compiles the
ground-truth program and randomizes that. Most mutants do not share the
ground-truth architecture (only 12-39% match its parameter count), so that
comparison measures architecture and initialization together and hands the
baseline the correct structure for free.

This sweep instead loads each mutant and randomizes that same architecture, so
the only thing that differs from the GBPR repair run is the starting point. Each
selected mutant already has a completed buggy-init result in saved_data/, giving
a paired comparison on identical jobs.

Usage
-----
    python launch_paired_init_control.py                      # 20 mutants x 5 seeds x 4 programs
    python launch_paired_init_control.py --subset-size 10 --seeds 0,1,2

Results land in {output-root}/{program}/job_{job_id}/seed_{seed}/cross_entropy_loss/
The selected mutant ids are written to {output-root}/selected_jobs.json so the
paired comparison is reproducible.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import submitit

PROGRAMS = ["hist", "most_freq", "reverse", "sort"]
LOSS_FN_NAME = "cross_entropy_loss"
N_EPOCHS = 10000
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
MUTATIONS_PATH = Path(__file__).parent.parent / "mutation/results/aggregated_mutations.json"
BUGGY_SAVED_DATA = Path(__file__).parent / "saved_data"


def select_mutants(program: str, subset_size: int, seed: int = 0) -> list[str]:
    """Stratified sample by mutation_order, restricted to mutants with a completed
    buggy-init result (so every selected job can be compared like-for-like)."""
    df = pd.read_json(MUTATIONS_PATH)
    buggy = df[df["execution_result"].apply(lambda r: r.get("status") == "BUGGY_MODEL")]
    buggy = buggy[buggy["program_name"] == program]

    saved_dir = BUGGY_SAVED_DATA / program / LOSS_FN_NAME
    completed = set()
    if saved_dir.exists():
        for d in saved_dir.iterdir():
            if (d / "test_results.json").exists():
                completed.add(d.name.replace("job_", ""))
    buggy = buggy[buggy["job_id"].isin(completed)]
    if buggy.empty:
        return []

    rng = np.random.default_rng(seed)
    orders = sorted(buggy["mutation_order"].unique())
    per_order = max(1, subset_size // len(orders))

    selected = []
    for order in orders:
        cands = buggy[buggy["mutation_order"] == order]["job_id"].tolist()
        selected.extend(rng.choice(cands, size=min(per_order, len(cands)), replace=False))
    # top up to subset_size from whatever remains, so we hit the requested count
    if len(selected) < subset_size:
        rest = [j for j in buggy["job_id"].tolist() if j not in set(selected)]
        if rest:
            extra = rng.choice(rest, size=min(subset_size - len(selected), len(rest)), replace=False)
            selected.extend(extra)
    return [str(j) for j in selected[:subset_size]]


def run_train_and_test(program_name: str, job_id: str, seed: int, output_dir: str,
                       init_scheme: str = "xavier"):
    repo_root = Path(__file__).parent.parent.parent
    container = repo_root / "container.sif"
    train_path = Path(__file__).parent / "train_mutations.py"
    test_path = Path(__file__).parent / "test_trained_mutations.py"
    full_out = Path(__file__).parent / output_dir

    common = [
        "--program_name", program_name,
        "--job_id", job_id,
        "--max_len", "10",
        "--loss_fn_name", LOSS_FN_NAME,
        "--random-init",
        "--random-init-source", "buggy",
        "--output_dir", str(full_out),
    ]
    train_cmd = ["apptainer", "exec", "--nv", str(container), "python", str(train_path),
                 "--n_epochs", str(N_EPOCHS), "--batch_size", str(BATCH_SIZE),
                 "--learning_rate", str(LEARNING_RATE), "--seed", str(seed),
                 "--no-store-trajectory", "--init-scheme", init_scheme] + common
    test_cmd = ["apptainer", "exec", "--nv", str(container), "python", str(test_path)] + common

    if not (full_out / "model.npy").exists():
        subprocess.run(train_cmd, check=True)
    subprocess.run(test_cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=str, default=None)
    ap.add_argument("--subset-size", type=int, default=20,
                    help="Mutants sampled per program (default 20).")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="Random-init seeds per mutant (default 5).")
    ap.add_argument("--subset-seed", type=int, default=0,
                    help="Seed for mutant selection; keep fixed for a reproducible pairing.")
    ap.add_argument("--init-scheme", type=str, default="xavier",
                    choices=["lecun", "xavier", "kaiming", "unit-normal"])
    ap.add_argument("--output-root", type=str, default="saved_data_paired_init")
    ap.add_argument("--timeout-min", type=int, default=4200)
    ap.add_argument("--array-parallelism", type=int, default=0,
                    help="Max concurrent array tasks. 0 (default) = unbounded: submit without a "
                         "throttle and let Slurm schedule as capacity frees up.")
    args = ap.parse_args()

    programs = args.programs.split(",") if args.programs else PROGRAMS
    seeds = [int(s) for s in args.seeds.split(",")]

    selection = {p: select_mutants(p, args.subset_size, args.subset_seed) for p in programs}
    out_root = Path(__file__).parent / args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "selected_jobs.json").write_text(json.dumps(
        {"subset_seed": args.subset_seed, "subset_size": args.subset_size,
         "seeds": seeds, "init_scheme": args.init_scheme, "selection": selection}, indent=2))

    for p, js in selection.items():
        print(f"{p}: {len(js)} mutants selected")

    executor = submitit.AutoExecutor(folder="paired_init_logs")
    exec_params = dict(
        name="RASPING-PAIRED-INIT", nodes=1, timeout_min=args.timeout_min,
        slurm_additional_parameters={"reservation": "1g.10gb"},
    )
    if args.array_parallelism > 0:
        exec_params["slurm_array_parallelism"] = args.array_parallelism
    executor.update_parameters(**exec_params)

    jobs = []
    with executor.batch():
        for program_name, job_ids in selection.items():
            for job_id in job_ids:
                for seed in seeds:
                    od = f"{args.output_root}/{program_name}/job_{job_id}/seed_{seed}/{LOSS_FN_NAME}"
                    if (Path(__file__).parent / od / "test_results.json").exists():
                        continue
                    jobs.append(executor.submit(
                        run_train_and_test, program_name, job_id, seed, od, args.init_scheme))

    total = sum(len(v) for v in selection.values()) * len(seeds)
    print(f"\nSubmitted {len(jobs)} jobs ({total} total combinations, "
          f"{total - len(jobs)} already complete).")
    print(f"Selection written to {out_root / 'selected_jobs.json'}")


if __name__ == "__main__":
    main()
