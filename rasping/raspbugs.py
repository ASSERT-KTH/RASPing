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

    def compile(self, max_len: int = MAX_LEN) -> "Model":
        """Compile this mutant into a Transformer (`Model`)."""
        from rasping.model import Model

        p = programs.get(self.program)
        sop = programs.build_sop(self.source, p.name, max_len, f"mutation_{p.name}_{self.job_id}")
        return Model(sop, p.vocab, max_len, p.name)


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
