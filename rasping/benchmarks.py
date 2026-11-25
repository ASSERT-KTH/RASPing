"""The two mutant benchmarks and their manifests.

A benchmark is a manifest (`<dir>/<name>.jsonl.gz`, one record per mutant: job_id, program,
mutation_order, operators, source, diff, oracle accuracy) plus an architecture index
(`<dir>/architectures.csv`). Both are produced by the generation pipelines in `raspbugs/`
and `tracrbench/` (see their READMEs).

    RASPBUGS     847 mutants of hist, most_freq, reverse and sort (`rasping.raspbugs`)
    TRACRBENCH   mutants of the included TracrBench programs (`rasping.tracrbench`)

Job ids are unique across benchmarks; `get_mutant(job_id)` finds a mutant in either.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from rasping import programs
from rasping.config import MAX_LEN, RASPBUGS_DIR, TRACRBENCH_DIR

if TYPE_CHECKING:
    from rasping.model import Model


@dataclass(frozen=True)
class Mutant:
    job_id: str
    program: str
    mutation_order: int
    operators: tuple
    source: str
    diff: str
    oracle_accuracy_50: float | None = None
    mutations: list = field(default_factory=list, compare=False, hash=False)
    benchmark: str = "raspbugs"

    def compile(self, max_len: int = MAX_LEN, **sharpness) -> Model:
        """Compile this mutant into a Transformer (`Model`).

        `sharpness` (`mlp_exactness`, `softmax_coldness`) is passed to `Model`; the defaults
        (the program's `mlp_exactness`, Tracr's coldness) compile the program exactly.
        """
        from rasping.model import Model

        p = programs.get(self.program)
        sop = programs.build_sop(self.source, p.name, max_len, f"mutation_{p.name}_{self.job_id}")
        return Model(sop, p.vocab, max_len, p.name, **{"mlp_exactness": p.mlp_exactness, **sharpness})


class Benchmark:
    def __init__(self, name: str, root: Path):
        self.name = name
        self.manifest = root / f"{name}.jsonl.gz"
        self.architectures_path = root / "architectures.csv"

    def __repr__(self):
        return f"Benchmark({self.name!r}, {len(self._mutants)} mutants)"

    @cached_property
    def _mutants(self) -> dict[str, Mutant]:
        out = {}
        if not self.manifest.exists():
            return out
        with gzip.open(self.manifest, "rt") as f:
            for line in f:
                r = json.loads(line)
                out[r["job_id"]] = Mutant(
                    job_id=r["job_id"],
                    program=programs.canonical(r["program"]),
                    mutation_order=int(r["mutation_order"]),
                    operators=tuple(r["operators"]),
                    source=r["source"],
                    diff=r["diff"],
                    oracle_accuracy_50=r.get("oracle_accuracy_50"),
                    mutations=r.get("mutations") or [],
                    benchmark=self.name,
                )
        return out

    def mutants(self, program: str | None = None, sort: bool = True) -> list[Mutant]:
        """The benchmark's mutants (optionally of one program).

        Sorted by (program, job_id), or in manifest order (the order the generation
        pipeline produced them) with `sort=False`.
        """
        ms = list(self._mutants.values())
        if program is not None:
            ms = [m for m in ms if m.program == programs.canonical(program)]
        return sorted(ms, key=lambda m: (m.program, m.job_id)) if sort else ms

    def programs(self) -> list[str]:
        return sorted({m.program for m in self._mutants.values()})

    def get(self, job_id: str) -> Mutant:
        return self._mutants[job_id]

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._mutants

    def stratified_sample(self, program: str, n: int, seed: int = 0) -> list[str]:
        """`n` job_ids of `program`, stratified by mutation order (manifest order, seeded).

        See `stratified_ids`; this is the selection rule of the v1 init control (20 per
        program, seed 0).
        """
        ms = self.mutants(program=program, sort=False)
        return stratified_ids([(m.job_id, m.mutation_order) for m in ms], n, seed)

    def architectures(self):
        """DataFrame: job_id, program, mutation_order, n_params, n_layers, signature."""
        import pandas as pd

        return pd.read_csv(self.architectures_path)

    def shape_twins(self, job_id: str, same_program: bool = True) -> list[str]:
        """Other mutants whose compiled parameters have exactly the same names and shapes."""
        t = self.architectures()
        row = t.loc[t["job_id"] == job_id].iloc[0]
        hit = (t["signature"] == row["signature"]) & (t["job_id"] != job_id)
        if same_program:
            hit &= t["program"] == row["program"]
        return sorted(t.loc[hit, "job_id"])


def stratified_ids(items: list[tuple[str, int]], n: int, seed: int = 0) -> list[str]:
    """`n` ids from `(id, mutation_order)` pairs (in a fixed order), stratified by order.

    n // (number of orders) per order (fewer if an order has fewer), drawn with
    `default_rng(seed)`, then topped up at random from the rest.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    orders = sorted({o for _, o in items})
    per_order = max(1, n // len(orders)) if orders else 0
    selected = []
    for o in orders:
        cands = [j for j, oo in items if oo == o]
        selected.extend(rng.choice(cands, size=min(per_order, len(cands)), replace=False))
    if len(selected) < n:  # top up from the remainder
        chosen = set(selected)
        rest = [j for j, _ in items if j not in chosen]
        if rest:
            selected.extend(rng.choice(rest, size=min(n - len(selected), len(rest)), replace=False))
    return [str(j) for j in selected[:n]]


RASPBUGS = Benchmark("raspbugs", RASPBUGS_DIR)
TRACRBENCH = Benchmark("tracrbench", TRACRBENCH_DIR)
BENCHMARKS = {b.name: b for b in (RASPBUGS, TRACRBENCH)}


def get_mutant(job_id: str) -> Mutant:
    """A mutant of either benchmark."""
    for b in BENCHMARKS.values():
        if job_id in b:
            return b.get(job_id)
    raise KeyError(f"{job_id} is in no benchmark")
