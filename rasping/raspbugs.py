"""Load the RaspBugs benchmark.

The manifest `raspbugs/raspbugs.jsonl.gz` has one record per benchmark mutant: job_id,
program, mutation_order, operators, source (the mutated program), diff and the oracle's
accuracy. RaspBugs is 847 buggy mutants of four programs (hist, most_freq, reverse,
sort). A mutant is included if the oracle labels it buggy and its compiled output
vocabulary can represent the correct outputs (docs/FINDINGS.md F6, F35). The manifest is
produced by the generation pipeline in `raspbugs/` (see its README).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from rasping import programs
from rasping.config import MAX_LEN, RASPBUGS_DIR

if TYPE_CHECKING:
    from rasping.model import Model

MANIFEST = RASPBUGS_DIR / "raspbugs.jsonl.gz"
#: Compiled architecture of every mutant (built by `raspbugs/architectures.py`).
ARCHITECTURES = RASPBUGS_DIR / "architectures.csv"


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

    def compile(self, max_len: int = MAX_LEN, **sharpness) -> "Model":
        """Compile this mutant into a Transformer (`Model`).

        `sharpness` (`mlp_exactness`, `softmax_coldness`) is passed to `Model`; the defaults
        compile the program exactly.
        """
        from rasping.model import Model

        p = programs.get(self.program)
        sop = programs.build_sop(self.source, p.name, max_len, f"mutation_{p.name}_{self.job_id}")
        return Model(sop, p.vocab, max_len, p.name, **sharpness)


@lru_cache(maxsize=1)
def _load() -> dict[str, Mutant]:
    out = {}
    with gzip.open(MANIFEST, "rt") as f:
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
            )
    return out


def mutants(program: str | None = None, sort: bool = True) -> list[Mutant]:
    """The benchmark's mutants (optionally of one program).

    Sorted by (program, job_id), or in manifest order (the order the generation
    pipeline produced them) with `sort=False`.
    """
    ms = list(_load().values())
    if program is not None:
        ms = [m for m in ms if m.program == programs.canonical(program)]
    return sorted(ms, key=lambda m: (m.program, m.job_id)) if sort else ms


def stratified_sample(program: str, n: int, seed: int = 0) -> list[str]:
    """`n` job_ids of `program`, stratified by mutation order (manifest order, seeded).

    n // (number of orders) per order (fewer if an order has fewer), topped up at random
    from the rest. This is the selection rule of the v1 init control (20 per program, seed 0).
    """
    import numpy as np

    ms = mutants(program=program, sort=False)
    rng = np.random.default_rng(seed)
    orders = sorted({m.mutation_order for m in ms})
    per_order = max(1, n // len(orders))
    selected = []
    for o in orders:
        cands = [m.job_id for m in ms if m.mutation_order == o]
        selected.extend(rng.choice(cands, size=min(per_order, len(cands)), replace=False))
    if len(selected) < n:  # top up from the remainder
        rest = [m.job_id for m in ms if m.job_id not in set(selected)]
        if rest:
            selected.extend(rng.choice(rest, size=min(n - len(selected), len(rest)), replace=False))
    return [str(j) for j in selected[:n]]


def architectures():
    """DataFrame: job_id, program, mutation_order, n_params, n_layers, signature."""
    import pandas as pd

    return pd.read_csv(ARCHITECTURES)


def shape_twins(job_id: str, same_program: bool = True) -> list[str]:
    """Other mutants whose compiled parameters have exactly the same names and shapes.

    Twins are almost always mutants of the same program. A few most_freq and reverse
    mutants share shapes across programs (7 mutants, 2 signatures); `same_program=False`
    includes those.
    """
    t = architectures()
    row = t.loc[t["job_id"] == job_id].iloc[0]
    hit = (t["signature"] == row["signature"]) & (t["job_id"] != job_id)
    if same_program:
        hit &= t["program"] == row["program"]
    return sorted(t.loc[hit, "job_id"])


def get(job_id: str) -> Mutant:
    return _load()[job_id]


def compile_mutant(job_id: str, max_len: int = MAX_LEN) -> "Model":
    return get(job_id).compile(max_len)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("Usage: python -m rasping.raspbugs <job_id>   (print a mutant)")
    m = get(sys.argv[1])
    print(f"job_id: {m.job_id}\nprogram: {m.program}\nmutation_order: {m.mutation_order}\n"
          f"operators: {', '.join(m.operators)}\n\n{m.diff}")
