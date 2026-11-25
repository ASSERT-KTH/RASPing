"""Load the TracrBench benchmark: mutants of the included TracrBench programs.

Same API as `rasping.raspbugs`, over `tracrbench/tracrbench.jsonl.gz` (built by the
pipeline in `tracrbench/`, see its README and `config.TRACRBENCH` for the rules).
Programs are `programs.tracrbench_programs()`; each has its own specification in
`data/tracrbench/`.
"""

from __future__ import annotations

from rasping.benchmarks import TRACRBENCH, Mutant  # noqa: F401  (re-exported)

MANIFEST = TRACRBENCH.manifest
ARCHITECTURES = TRACRBENCH.architectures_path

mutants = TRACRBENCH.mutants
get = TRACRBENCH.get
stratified_sample = TRACRBENCH.stratified_sample
architectures = TRACRBENCH.architectures
shape_twins = TRACRBENCH.shape_twins
