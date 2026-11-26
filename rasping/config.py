"""Single source of truth for paths, program lists and experiment constants.

Every script imports constants from here instead of redefining them. If you need a
new constant that more than one script uses, add it here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------- paths

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"  # input-output specifications (train/val/test JSONL)
RASPBUGS_DIR = REPO_ROOT / "raspbugs"  # benchmark generation pipeline + manifest
TRACRBENCH_DIR = REPO_ROOT / "tracrbench"  # second benchmark: mutants of TracrBench programs
RESULTS_DIR = REPO_ROOT / "results"  # tidy, versioned result tables (tracked)
ARTIFACTS_DIR = REPO_ROOT / "paper_artifacts"  # generated tables / figures (tracked)
RUNS_DIR = REPO_ROOT / "runs"  # raw run outputs incl. weights (gitignored)

# --------------------------------------------------------------------------- programs

#: Canonical names of the RaspBugs base programs (as used in the benchmark and results).
#: The v1 paper also had shuffle_dyck and shuffle_dyck2; they were removed (FINDINGS F35).
PROGRAMS = ("hist", "most_freq", "reverse", "sort")

#: Maximum input length (excluding BOS) of the specifications and compiled models.
MAX_LEN = 10

#: Tracr's MLP sharpness (`mlp_exactness`) for RaspBugs programs (Tracr's default is 100;
#: v1 used 120). TracrBench programs need more to compile exactly (see `TRACRBENCH`).
MLP_EXACTNESS = 120

# --------------------------------------------------------------------------- TracrBench


@dataclass(frozen=True)
class TracrBenchRules:
    """How the TracrBench benchmark is built (fixed Nov 25, 2026, before any mutant was run).

    Source: github.com/HannesThurnherr/TracrBench (Thurnherr & Riesen, 2024), `eval_data.json`
    at `commit`: 101 programs with RASP code and a ground-truth Python function. A program
    is included if
      1. its RASP code runs in TracrBench's own environment (`from tracr.rasp import rasp`),
      2. compiled with `mlp_exactness`, it reproduces the ground truth on every test example,
      3. it is not trivial: neither one constant output sequence nor copying the input
         reaches `max_trivial_accuracy` exact-match accuracy on the test split,
      4. it is not a duplicate: of programs with the same source up to names, comments and
         docstrings (`programs.canonical_source`), only the first by name is kept.
    There is no size limit. (A 1.2M-parameter cap, the largest RaspBugs model, was dropped
    on Nov 25, 2026, before any mutant was generated: it had no scientific basis. Rule 4 was
    added on Nov 25, 2026, after mutants were generated but before the manifest was built or
    any repair was run. It applies in step 0, and each program's mutants are generated and
    sampled independently, so the result equals applying it before generation.)
    Mutants: buggy and representable (as RaspBugs), then at most `mutants_per_program` per
    program, stratified by mutation order (seed `sample_seed`).
    """

    repo: str = "https://github.com/HannesThurnherr/TracrBench"
    commit: str = "48abdd9749b568af1015bcee72ff19140063cc5e"
    vocab: tuple = tuple(range(10))
    mlp_exactness: float = 1000  # TracrBench's own setting; 3 programs are wrong at 120
    max_trivial_accuracy: float = 0.5
    mutants_per_program: int = 20
    sample_seed: int = 0


TRACRBENCH = TracrBenchRules()

# --------------------------------------------------------------------------- labels

#: Name of the method in figure legends and table headers. The v1 paper called it
#: "GBPR"; the resubmission is expected to rename it (AE decision, Nov 2026).
METHOD_LABEL = "GBPR"

# --------------------------------------------------------------------------- evaluation

#: A program counts as repaired if its exact-sequence accuracy on held-out data
#: is at least this value.
THRESHOLD = 0.99

# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class TrainConfig:
    """Gradient-based repair hyperparameters (paper §5.1).

    These are the values the v1 (TMLR) runs actually used. Note the early-stopping
    patience: the v1 paper text says 10, the runs used 50 (docs/FINDINGS.md F2).
    """

    n_epochs: int = 10_000
    batch_size: int = 256
    learning_rate: float = 1e-4
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    val_every: int = 10  # validation accuracy is computed every `val_every` epochs
    grad_clip_norm: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)
