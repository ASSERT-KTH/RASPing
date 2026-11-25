"""Load the RaspBugs benchmark.

The manifest `raspbugs/raspbugs.jsonl.gz` has one record per benchmark mutant: job_id,
program, mutation_order, operators, source (the mutated program), diff and the oracle's
accuracy. RaspBugs is 847 buggy mutants of four programs (hist, most_freq, reverse,
sort). A mutant is included if the oracle labels it buggy and its compiled output
vocabulary can represent the correct outputs (docs/FINDINGS.md F6, F35). The manifest is
produced by the generation pipeline in `raspbugs/` (see its README).

This module is `benchmarks.RASPBUGS` as functions; `rasping.tracrbench` is the same for
the TracrBench mutants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rasping.benchmarks import RASPBUGS, Mutant  # noqa: F401  (re-exported)
from rasping.config import MAX_LEN

if TYPE_CHECKING:
    from rasping.model import Model

MANIFEST = RASPBUGS.manifest
#: Compiled architecture of every mutant (built by `raspbugs/architectures.py`).
ARCHITECTURES = RASPBUGS.architectures_path

mutants = RASPBUGS.mutants
get = RASPBUGS.get
stratified_sample = RASPBUGS.stratified_sample
architectures = RASPBUGS.architectures


def shape_twins(job_id: str, same_program: bool = True) -> list[str]:
    """Other mutants whose compiled parameters have exactly the same names and shapes.

    Twins are almost always mutants of the same program. A few most_freq and reverse
    mutants share shapes across programs (7 mutants, 2 signatures); `same_program=False`
    includes those.
    """
    return RASPBUGS.shape_twins(job_id, same_program)


def compile_mutant(job_id: str, max_len: int = MAX_LEN) -> "Model":
    return get(job_id).compile(max_len)


if __name__ == "__main__":
    import sys

    from rasping.benchmarks import get_mutant

    if len(sys.argv) != 2:
        sys.exit("Usage: python -m rasping.raspbugs <job_id>   (print a mutant of either benchmark)")
    m = get_mutant(sys.argv[1])
    print(f"job_id: {m.job_id}\nbenchmark: {m.benchmark}\nprogram: {m.program}\n"
          f"mutation_order: {m.mutation_order}\noperators: {', '.join(m.operators)}\n\n{m.diff}")
