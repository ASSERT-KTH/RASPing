"""Weight statistics of the models used in the initialization control.

Reports, per base program and over the 20 mutants selected for the paired control:
the fraction of parameters that are exactly zero, the standard deviation of the
model's full parameter vector, the largest weight magnitude, and the standard
deviation a fan-in-scaled (Xavier) initialization would use on the same tensors.
These are the numbers in the appendix table on initialization.

Two sets of weights are reported and must not be confused. "compiled" is the
buggy starting point, i.e. what Tracr produces and what GBPR initializes from.
"repaired" is the same mutants' weights once GBPR has finished, loaded from
saved_data/. Both blocks use the same statistics so that the columns of the
appendix table are directly comparable: optimization fills in a large share of
the compiled zeros but leaves the scale of the parameter vector unchanged.

Usage:
    python compiled_weight_stats.py
"""
import json
from pathlib import Path

import numpy as np

from experiments.mutation.load_mutations import load_buggy_models

HERE = Path(__file__).parent
PROGRAMS = ["hist", "most_freq", "reverse", "sort"]
MAX_LEN = 10


def vector_stats(params: dict) -> tuple[float, float, float]:
    """Zero fraction, standard deviation and max magnitude of the full parameter vector."""
    flat = np.concatenate(
        [np.asarray(t).ravel() for layer in params.values() for t in layer.values()]
    )
    return float((flat == 0).mean()), float(flat.std()), float(np.abs(flat).max())


def xavier_stds(params: dict) -> list[float]:
    """The std a fan-in-scaled init would use on each 2-D tensor of this model."""
    return [
        float(np.sqrt(2.0 / (a.shape[0] + a.shape[1])))
        for layer in params.values()
        for a in map(np.asarray, layer.values())
        if a.ndim == 2
    ]


def report(label: str, program: str, zeros, stds, max_abs, xavier=None) -> None:
    line = (
        f"  {program:<10s} {label:<9s} zero {min(zeros):.3f}-{max(zeros):.3f}  "
        f"std {min(stds):.2f}-{max(stds):.2f}  max|w| {max(max_abs):.2f}"
    )
    if xavier:
        line += f"  xavier std {min(xavier):.3f}-{max(xavier):.3f}"
    print(line, flush=True)


def main():
    selection = json.load(
        open(HERE / "saved_data_paired_init" / "selected_jobs.json")
    )["selection"]

    for program in PROGRAMS:
        c_zeros, c_stds, c_max, xavier = [], [], [], []
        r_zeros, r_stds, r_max = [], [], []
        x_zeros, x_stds, x_max = [], [], []

        for job_id in selection[program]:
            model = load_buggy_models(
                max_length=MAX_LEN, program_name=program, job_id=job_id
            )[job_id]
            z, s, m = vector_stats(model.model.params)
            c_zeros.append(z)
            c_stds.append(s)
            c_max.append(m)
            xavier.extend(xavier_stds(model.model.params))

            path = (
                HERE / "saved_data" / program / "cross_entropy_loss"
                / f"job_{job_id}" / "model.npy"
            )
            if not path.exists():
                continue
            z, s, m = vector_stats(np.load(path, allow_pickle=True).item())
            r_zeros.append(z)
            r_stds.append(s)
            r_max.append(m)

            for seed_dir in sorted((HERE / "saved_data_paired_init" / program
                                    / f"job_{job_id}").glob("seed_*")):
                seed_path = seed_dir / "cross_entropy_loss" / "model.npy"
                if not seed_path.exists():
                    continue
                z, s, m = vector_stats(np.load(seed_path, allow_pickle=True).item())
                x_zeros.append(z)
                x_stds.append(s)
                x_max.append(m)

        report("compiled", program, c_zeros, c_stds, c_max, xavier)
        report("repaired", program, r_zeros, r_stds, r_max)
        report("random", program, x_zeros, x_stds, x_max)


if __name__ == "__main__":
    main()
