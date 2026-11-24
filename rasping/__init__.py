"""RASPing: repairing Tracr-compiled RASP programs by gradient descent.

Package layout:
    config      paths, program lists, thresholds, training defaults
    programs    registry of the base RASP programs (vocabularies, factories, aliases)
    generators  ground-truth input/output generators for each program
    data        specification datasets: generate, split, save, load, encode
    model       `Model`: a compiled RASP program as a Transformer (Tracr + Haiku)
    losses      correctness losses
    metrics     exact-match accuracy, pass rates, Dyck balanced metrics
    train       `Trainer` (Adam + early stopping) used for gradient-based repair
    raspbugs    load the RaspBugs benchmark (mutant sources -> `Model`)
    repair      end-to-end repair of one mutant (the method), writes one run dir
    cluster     SLURM (submitit) + apptainer helpers for launchers
    tables      LaTeX table writer for paper artifacts

JAX is configured (x64, float32 matmuls) by `rasping._jax_setup`, imported by the modules
that use JAX (`model`, `losses`, `train`). The symbolic parts (programs, data loading,
raspbugs manifest, baselines) do not import JAX, so the baselines' worker processes can
be forked safely (F32).
"""
