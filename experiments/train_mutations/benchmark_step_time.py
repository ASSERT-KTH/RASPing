"""
Benchmark GBPR training step time for one representative job per program.

Runs warmup_epochs to trigger JIT compilation, then times measure_epochs
and records seconds_per_epoch and seconds_per_val_step.

Run via launch_benchmark_step_time.py on the cluster, or directly:
    python benchmark_step_time.py --program_name hist --job_id <id> --output_path out.json
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import jax

from src.trainer import Trainer
from src.functions import load_dataset, encodeAndPadData
from src.loss import cross_entropy_loss
from experiments.mutation.load_mutations import load_buggy_models


VAL_STEP = 10  # Must match valStep used in actual training


def benchmark(
    program_name: str,
    job_id: str,
    max_len: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    warmup_epochs: int = 50,
    measure_epochs: int = 200,
    output_path: str = None,
):
    # Load data
    program_name_key = program_name
    if program_name == "most_freq":
        program_name_key = "most-freq"
    elif program_name == "shuffle_dyck":
        program_name_key = "shuffle_dyck1"

    data_path = str(Path(__file__).parent.resolve() / "../../data/")
    train_dataset = load_dataset(data_path, program_name_key, split_name="train")
    val_dataset = load_dataset(data_path, program_name_key, split_name="val")

    # Load both models upfront so load time doesn't pollute timing
    model1 = load_buggy_models(max_length=max_len, program_name=program_name, job_id=job_id)[job_id]
    model2 = load_buggy_models(max_length=max_len, program_name=program_name, job_id=job_id)[job_id]

    X_train, Y_train = encodeAndPadData(
        train_dataset, model1.raspFunction, model1.inputs, max_len
    )
    X_val, Y_val = encodeAndPadData(
        val_dataset, model1.raspFunction, model1.inputs, max_len
    )

    def make_trainer(model_obj, n_epochs):
        return Trainer(
            model=model_obj,
            X_train=X_train,
            Y_train=Y_train,
            loss_fn=cross_entropy_loss,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            X_val=X_val,
            Y_val=Y_val,
            valStep=VAL_STEP,
            use_wandb=False,
        )

    # Warmup: triggers JIT compilation
    print(f"[{program_name}/{job_id}] Warming up ({warmup_epochs} epochs)...")
    make_trainer(model1, warmup_epochs).train()

    # Measurement: fresh model, JIT already compiled
    print(f"[{program_name}/{job_id}] Measuring ({measure_epochs} epochs)...")
    measure_trainer = make_trainer(model2, measure_epochs)

    t0 = time.perf_counter()
    measure_trainer.train()
    jax.effects_barrier()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    seconds_per_epoch = elapsed / measure_epochs
    seconds_per_val_step = seconds_per_epoch * VAL_STEP

    result = {
        "program_name": program_name,
        "job_id": job_id,
        "warmup_epochs": warmup_epochs,
        "measure_epochs": measure_epochs,
        "elapsed_s": elapsed,
        "seconds_per_epoch": seconds_per_epoch,
        "val_step": VAL_STEP,
        "seconds_per_val_step": seconds_per_val_step,
    }

    print(json.dumps(result, indent=2))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GBPR training step time")
    parser.add_argument("--program_name", type=str, required=True)
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--measure_epochs", type=int, default=200)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    benchmark(
        program_name=args.program_name,
        job_id=args.job_id,
        max_len=args.max_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        measure_epochs=args.measure_epochs,
        output_path=args.output_path,
    )
