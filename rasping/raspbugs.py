"""Load the RaspBugs benchmark.

The benchmark manifest `raspbugs/raspbugs_<version>.jsonl.gz` has one record per buggy
mutant: job_id, program, mutation_order, operators, source (the mutated program), diff,
and flags. It is produced by the generation pipeline in `raspbugs/` (see its README).

Subsets:
    "all"   the benchmark proper (v1: the 1,466 mutants of the TMLR submission)
    "main"  the sequence-to-sequence programs of the main evaluation (v1: 847)
    "dyck"  the two Dyck programs
    "buggy" every mutant labelled BUGGY_MODEL by the oracle (v1: 1,773; see F6)
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rasping import programs
from rasping.config import DYCK_PROGRAMS, MAIN_PROGRAMS, MAX_LEN, RASPBUGS_DIR
from rasping.model import Model

DEFAULT_VERSION = "v1"
SUBSETS = ("all", "main", "dyck", "buggy")


@dataclass(frozen=True)
class Mutant:
    job_id: str
    program: str
    mutation_order: int
    operators: tuple
    source: str
    diff: str
    in_benchmark: bool
    oracle_accuracy_50: float | None = None
    mutations: list = field(default_factory=list, compare=False, hash=False)

    def compile(self, max_len: int = MAX_LEN) -> Model:
        """Compile this mutant into a Transformer (`Model`)."""
        p = programs.get(self.program)
        sop = programs.build_sop(self.source, p.name, max_len, f"mutation_{p.name}_{self.job_id}")
        return Model(sop, p.vocab, max_len, p.name)


def manifest_path(version: str = DEFAULT_VERSION) -> Path:
    return RASPBUGS_DIR / f"raspbugs_{version}.jsonl.gz"


@lru_cache(maxsize=4)
def _load(version: str) -> dict[str, Mutant]:
    out = {}
    with gzip.open(manifest_path(version), "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["job_id"]] = Mutant(
                job_id=r["job_id"],
                program=programs.canonical(r["program"]),
                mutation_order=int(r["mutation_order"]),
                operators=tuple(r["operators"]),
                source=r["source"],
                diff=r["diff"],
                in_benchmark=bool(r.get(f"in_raspbugs_{version}", True)),
                oracle_accuracy_50=r.get("oracle_accuracy_50"),
                mutations=r.get("mutations") or [],
            )
    return out


def mutants(
    subset: str = "all",
    program: str | None = None,
    version: str = DEFAULT_VERSION,
) -> list[Mutant]:
    """Mutants of a subset (optionally of one program), sorted by (program, job_id)."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    ms = list(_load(version).values())
    if subset != "buggy":
        ms = [m for m in ms if m.in_benchmark]
    if subset == "main":
        ms = [m for m in ms if m.program in MAIN_PROGRAMS]
    elif subset == "dyck":
        ms = [m for m in ms if m.program in DYCK_PROGRAMS]
    if program is not None:
        ms = [m for m in ms if m.program == programs.canonical(program)]
    return sorted(ms, key=lambda m: (m.program, m.job_id))


def get(job_id: str, version: str = DEFAULT_VERSION) -> Mutant:
    return _load(version)[job_id]


def compile_mutant(job_id: str, max_len: int = MAX_LEN, version: str = DEFAULT_VERSION) -> Model:
    return get(job_id, version).compile(max_len)
