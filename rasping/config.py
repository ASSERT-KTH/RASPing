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
RESULTS_DIR = REPO_ROOT / "results"  # tidy, versioned result tables (tracked)
ARTIFACTS_DIR = REPO_ROOT / "paper_artifacts"  # generated tables / figures (tracked)
RUNS_DIR = REPO_ROOT / "runs"  # raw run outputs incl. weights (gitignored)

# --------------------------------------------------------------------------- programs

#: Canonical names of the RaspBugs base programs (as used in the benchmark and results).
#: The v1 paper also had shuffle_dyck and shuffle_dyck2; they were removed (FINDINGS F35).
PROGRAMS = ("hist", "most_freq", "reverse", "sort")

#: Maximum input length (excluding BOS) of the specifications and compiled models.
MAX_LEN = 10

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
