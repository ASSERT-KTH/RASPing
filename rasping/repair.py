"""Gradient-based repair of one benchmark mutant (RaspBugs or TracrBench), end to end.

    compile mutant -> (optionally re-initialise weights) -> encode specification
    -> train with early stopping on validation loss -> evaluate on the test split

Each run writes one directory:
    run.json      config, provenance and results (see `RunRecord`)
    curves.npz    train_losses, train_accs, val_losses (per epoch); val_accs and
                  selected_test_accs (test accuracy of the best-val-loss params) every val_every
    model.npy     final (best) parameters                      [gitignored]
    trajectory.pkl parameter trajectory, if requested          [gitignored]

CLI (used by all launchers):
    python -m rasping.repair --job-id <id> --out runs/main_repair/<id> [options]
"""

from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import click
import numpy as np

from rasping import data
from rasping.benchmarks import get_mutant
from rasping.config import MAX_LEN, REPO_ROOT, TrainConfig
from rasping.losses import LOSSES
from rasping.model import INIT_SCHEMES, SOFTMAX_COLDNESS, Model
from rasping.train import Trainer

#: Starting points, all in the mutant's own compiled architecture:
#:   compiled        the mutant's Tracr weights (the method)
#:   lecun | xavier | kaiming | unit-normal   random re-initialisation (`Model.set_random_weights`)
#:   shuffled        compiled values permuted within each tensor: scale and sparsity kept,
#:                   structure destroyed
#:   rescaled        compiled structure at Xavier scale, biases zeroed
#:   soft:<c>        compiled with attention coldness c and the program's MLP exactness scaled
#:                   by c/100 (exact: 100); softer weights that approximate the mutant
#:   from:<job_id>   the compiled weights of another mutant with identical parameter shapes
#:                   (usually of the same program; see `Benchmark.shape_twins`)
INITS = ("compiled", *INIT_SCHEMES, "shuffled", "rescaled", "soft:<c>", "from:<job_id>")


def check_init(init: str) -> str:
    kind, sep, arg = init.partition(":")
    ok = (not sep and kind in ("compiled", *INIT_SCHEMES, "shuffled", "rescaled")) or (
        sep and kind == "soft" and arg.replace(".", "", 1).isdigit() and float(arg) > 0) or (
        sep and kind == "from" and len(arg) == 32)
    if not ok:
        raise ValueError(f"init {init!r} is not one of {INITS}")
    return init


def _start_model(mutant, init: str, compiled: Model, seed: int) -> Model:
    """The model training starts from (`compiled` itself for init="compiled")."""
    kind, _, arg = init.partition(":")
    if kind == "compiled":
        return compiled
    if kind == "soft":
        c = float(arg)
        exactness = c * compiled.mlp_exactness / SOFTMAX_COLDNESS
        return mutant.compile(MAX_LEN, softmax_coldness=c, mlp_exactness=exactness)
    model = mutant.compile(MAX_LEN)  # a fresh copy whose weights we replace
    model.set_prng_key(seed)
    if kind in INIT_SCHEMES:
        model.set_random_weights(kind)
    elif kind == "shuffled":
        model.set_shuffled_weights()
    elif kind == "rescaled":
        model.set_rescaled_weights()
    elif kind == "from":
        if arg == mutant.job_id:
            raise ValueError(f"from:{arg} must be another mutant")
        model.set_weights(get_mutant(arg).compile(MAX_LEN).params)  # asserts identical shapes
    return model


def _agreement(a: Model, b: Model, X) -> float:
    """Fraction of inputs on which two models predict the same full output sequence."""
    import jax.numpy as jnp

    pa = jnp.argmax(a.forward(X).unembedded_output, -1)
    pb = jnp.argmax(b.forward(X).unembedded_output, -1)
    mask = jnp.ones_like(X).at[:, 0].set(0)
    mask = jnp.where(X != a.pad_token, mask, 0)
    return float(jnp.mean(jnp.all((pa == pb) | (mask == 0), axis=-1)))


@dataclass
class RunRecord:
    job_id: str
    program: str
    mutation_order: int
    init: str
    seed: int
    n_train: int
    n_val: int
    n_steps_budget: int | None
    config: dict
    # results
    status: str = "ok"
    test_accuracy: float = float("nan")
    test_samples: int = 0
    initial_test_accuracy: float = float("nan")  # the compiled mutant's (the bug's) accuracy
    start_test_accuracy: float = float("nan")  # accuracy of the weights training starts from
    start_agreement: float = float("nan")  # fraction of test inputs where the start = compiled mutant
    n_epochs: int = 0
    stopped_early: bool = False
    diverged: bool = False
    best_step: int | None = None
    final_step: int = 0
    best_val_acc: float = float("nan")
    wall_time_s: float = float("nan")
    epoch_end_times_s: list = field(default_factory=list)
    # provenance
    git_commit: str = ""
    device: str = ""  # JAX device the run trained on, e.g. "cuda:0" or "cpu:0"
    xla_flags: str = ""  # XLA_FLAGS at run time; v2 GPU runs are deterministic (F39)


def _device() -> str:
    import jax

    d = jax.devices()[0]
    return f"{d.platform}:{d.id}"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def _subsample(X, Y, n: int | None, seed: int):
    if n is None or n >= len(X):
        return X, Y
    idx = np.random.default_rng(seed=seed).choice(len(X), n, replace=False)
    return X[idx], Y[idx]


def repair(
    job_id: str,
    out_dir: str | Path | None = None,
    config: TrainConfig = TrainConfig(),
    init: str = "compiled",
    seed: int = 42,
    n_train: int | None = None,
    n_val: int | None = None,
    n_steps: int | None = None,
    store_trajectory: bool = False,
    trajectory_store_interval: int = 50,
    save_weights: bool = True,
    wandb_project: str | None = None,
    progress: bool = True,
) -> RunRecord:
    """Repair one mutant and (if `out_dir`) write its run directory.

    `seed` sets the batch order (every init), the re-initialisation (non-compiled inits) and
    the `n_train`/`n_val` subsample of the specification. `n_steps` fixes the gradient-step
    budget by setting the epoch count (spec-size ablation).
    """
    check_init(init)
    mutant = get_mutant(job_id)
    compiled = mutant.compile(MAX_LEN)
    # Benchmark mutants can represent every target output by construction (F6, F35).
    missing = data.unrepresentable_outputs(mutant.program, compiled)
    if missing:
        raise ValueError(f"{job_id}: compiled output vocabulary lacks {sorted(missing)[:5]}")

    X, Y = data.encode(data.load(mutant.program, "train"), compiled)
    Xv, Yv = data.encode(data.load(mutant.program, "val"), compiled)
    Xt, Yt = data.encode(data.load(mutant.program, "test"), compiled)
    X, Y = _subsample(X, Y, n_train, seed)
    Xv, Yv = _subsample(Xv, Yv, n_val, seed)

    initial_test_acc = float(compiled.accuracy(Xt, Yt))
    model = _start_model(mutant, init, compiled, seed)
    start_test_acc = float(model.accuracy(Xt, Yt))
    start_agreement = 1.0 if model is compiled else _agreement(model, compiled, Xt)

    n_epochs = config.n_epochs
    if n_steps is not None:
        n_epochs = math.ceil(n_steps / max(1, len(X) // config.batch_size))

    rec = RunRecord(
        job_id=job_id, program=mutant.program, mutation_order=mutant.mutation_order,
        init=init, seed=seed, n_train=len(X), n_val=len(Xv), n_steps_budget=n_steps,
        config={**config.to_dict(), "n_epochs": n_epochs}, initial_test_accuracy=initial_test_acc,
        start_test_accuracy=start_test_acc, start_agreement=start_agreement,
        git_commit=_git_commit(), device=_device(), xla_flags=os.environ.get("XLA_FLAGS", ""),
    )

    wandb_run = None
    if wandb_project:
        import wandb

        wandb_run = wandb.init(project=wandb_project, name=f"{mutant.program}_{job_id}_{init}_s{seed}",
                               config=asdict(rec))

    trainer = Trainer(
        model, X, Y, LOSSES["cross_entropy_loss"],
        n_epochs=n_epochs, batch_size=config.batch_size, lr=config.learning_rate,
        X_val=Xv, Y_val=Yv, X_test=Xt, Y_test=Yt,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        val_every=config.val_every, grad_clip_norm=config.grad_clip_norm,
        store_trajectory=store_trajectory, trajectory_store_interval=trajectory_store_interval,
        wandb_run=wandb_run, progress=progress, seed=seed,
    )
    t0 = time.time()
    result = trainer.train()
    rec.wall_time_s = time.time() - t0

    rec.test_accuracy = float(model.accuracy(Xt, Yt))
    rec.test_samples = int(len(Xt))
    rec.n_epochs = len(result.train_losses)
    rec.stopped_early = result.stopped_early
    rec.diverged = result.diverged
    rec.best_step = result.best_step
    rec.final_step = result.final_step
    rec.best_val_acc = float(max(result.val_accs)) if result.val_accs else float("nan")
    rec.epoch_end_times_s = [round(float(t), 3) for t in result.epoch_end_times]
    if wandb_run is not None:
        wandb_run.summary["test_accuracy"] = rec.test_accuracy
        wandb_run.finish()

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "curves.npz", **{k: np.asarray(v) for k, v in result.history().items()})
        if save_weights:
            model.save(out / "model.npy")
        if store_trajectory:
            with open(out / "trajectory.pkl", "wb") as f:
                pickle.dump(result.trajectory, f)
        (out / "run.json").write_text(json.dumps(asdict(rec), indent=1, default=float))
    return rec


def _init_option(ctx, param, value):
    try:
        return check_init(value)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


@click.command()
@click.option("--job-id", required=True, help="Mutant id (RaspBugs or TracrBench)")
@click.option("--out", "out_dir", required=True, type=click.Path(), help="Run directory to write")
@click.option("--init", default="compiled", show_default=True, callback=_init_option,
              help=f"One of {', '.join(INITS)}")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Seed for batch order, re-initialisation and specification subsampling")
@click.option("--n-train", type=int, default=None, help="Subsample the training split")
@click.option("--n-val", type=int, default=None, help="Subsample the validation split")
@click.option("--n-steps", type=int, default=None, help="Fixed gradient-step budget")
@click.option("--n-epochs", type=int, default=TrainConfig.n_epochs, show_default=True)
@click.option("--patience", type=int, default=TrainConfig.early_stopping_patience, show_default=True)
@click.option("--store-trajectory", is_flag=True)
@click.option("--trajectory-interval", type=int, default=50, show_default=True)
@click.option("--no-weights", is_flag=True, help="Do not save model.npy")
@click.option("--wandb-project", default=None, help="Log to Weights & Biases")
def main(job_id, out_dir, init, seed, n_train, n_val, n_steps, n_epochs, patience,
         store_trajectory, trajectory_interval, no_weights, wandb_project):
    config = TrainConfig(n_epochs=n_epochs, early_stopping_patience=patience)
    rec = repair(job_id, out_dir, config, init, seed, n_train, n_val, n_steps,
                 store_trajectory, trajectory_interval, not no_weights, wandb_project)
    print(json.dumps({"job_id": rec.job_id, "test_accuracy": rec.test_accuracy,
                      "n_epochs": rec.n_epochs}))


if __name__ == "__main__":
    main()
