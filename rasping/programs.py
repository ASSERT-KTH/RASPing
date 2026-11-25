"""Registry of the base RASP programs of both benchmarks.

Each program has a canonical name (`most_freq`, ...), the file prefix of its
specification in `data/` (historically `most-freq` for most_freq), an input
vocabulary, and a factory that builds the RASP SOp from a module containing the
program's source (the reference source in `<benchmark>/source/` or a mutant's source).

RaspBugs programs (`config.PROGRAMS`) are defined here. TracrBench programs are read from
`tracrbench/programs.json` (written by `tracrbench/import_programs.py`); only the
included ones are registered (`tracrbench_programs()`). Their specifications live in
`data/tracrbench/` and are generated from TracrBench's ground-truth functions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable

from tracr.rasp import rasp

from rasping.config import MAX_LEN, MLP_EXACTNESS, RASPBUGS_DIR, TRACRBENCH, TRACRBENCH_DIR

SOURCE_DIR = RASPBUGS_DIR / "source"
#: TracrBench programs: one record per program of eval_data.json, included or not.
TRACRBENCH_PROGRAMS = TRACRBENCH_DIR / "programs.json"
#: Prepended to every TracrBench program, as TracrBench's own pipeline does.
TRACRBENCH_HEADER = "from tracr.rasp import rasp\n\n"


@dataclass(frozen=True)
class Program:
    name: str
    data_name: str
    vocab: tuple
    _factory: Callable[[ModuleType, int, tuple], rasp.SOp]
    benchmark: str = "raspbugs"
    mlp_exactness: float = MLP_EXACTNESS
    #: TracrBench: ground truth on a raw input list (a list, or one value for every position)
    ground_truth: Callable[[list], object] | None = None

    def build(self, module: ModuleType, max_len: int = MAX_LEN) -> rasp.SOp:
        """Build the SOp from a module that defines this program's `make_*` function."""
        return self._factory(module, max_len, self.vocab)

    @property
    def source_path(self) -> Path:
        root = RASPBUGS_DIR if self.benchmark == "raspbugs" else TRACRBENCH_DIR
        return root / "source" / f"{self.name}.py"

    def data_file(self, split: str) -> str:
        """Specification file of a split, relative to the data directory."""
        if self.benchmark == "raspbugs":
            return f"{self.data_name}_{split}.jsonl"
        return f"tracrbench/{self.data_name}_{split}.jsonl.gz"

    def target(self, inputs: list) -> list:
        """TracrBench: the ground-truth output sequence for `inputs` (no BOS)."""
        y = self.ground_truth(list(inputs))
        return list(y) if isinstance(y, (list, tuple)) else [y] * len(inputs)


_PROGRAMS = {
    "hist": Program("hist", "hist", ("a", "b", "c", "d"), lambda m, n, v: m.make_hist()),
    "sort": Program(
        "sort",
        "sort",
        # 0 is excluded: compiled sort fails on some inputs when 0 is in the vocabulary
        (1, 2, 3, 4, 5, 6),
        lambda m, n, v: m.make_sort(rasp.tokens, rasp.tokens, max_seq_len=n, min_key=min(v)),
    ),
    "reverse": Program(
        "reverse", "reverse", ("a", "b", "c", "d", "e"), lambda m, n, v: m.make_reverse(rasp.tokens)
    ),
    "most_freq": Program(
        "most_freq", "most-freq", (1, 2, 3, 4, 5), lambda m, n, v: m.make_sort_freq(n)
    ),
}

#: Historical names still found in data files and old result records.
ALIASES = {p.data_name: p.name for p in _PROGRAMS.values()}


def _ground_truth(source: str) -> Callable[[list], object]:
    ns: dict = {}
    exec(source, ns)
    return ns["fun"]


@lru_cache(maxsize=1)
def tracrbench_records() -> tuple[dict, ...]:
    """Every TracrBench program record (included or not), from `tracrbench/programs.json`."""
    if not TRACRBENCH_PROGRAMS.exists():
        return ()
    return tuple(json.loads(TRACRBENCH_PROGRAMS.read_text()))


@lru_cache(maxsize=1)
def _tracrbench() -> dict[str, Program]:
    out = {}
    for r in tracrbench_records():
        if not r["included"]:
            continue
        assert r["name"] not in _PROGRAMS, f"TracrBench program {r['name']} shadows a RaspBugs program"
        entry = r["entry"]
        out[r["name"]] = Program(
            r["name"], r["name"], TRACRBENCH.vocab, lambda m, n, v, entry=entry: getattr(m, entry)(),
            benchmark="tracrbench", mlp_exactness=TRACRBENCH.mlp_exactness,
            ground_truth=_ground_truth(r["ground_truth"]))
    return out


def canonical(name: str) -> str:
    """Map any historical program name (e.g. `most-freq`) to the canonical one."""
    name = ALIASES.get(name, name)
    if name not in _PROGRAMS and name not in _tracrbench():
        raise KeyError(f"Unknown program {name!r}; known: {sorted(_PROGRAMS)} and "
                       f"{len(_tracrbench())} TracrBench programs")
    return name


def get(name: str | Program) -> Program:
    if isinstance(name, Program):  # e.g. a TracrBench program being imported
        return name
    name = canonical(name)
    return _PROGRAMS[name] if name in _PROGRAMS else _tracrbench()[name]


def all_programs() -> list[Program]:
    """The RaspBugs programs (see `tracrbench_programs` for the others)."""
    return list(_PROGRAMS.values())


def tracrbench_programs() -> list[Program]:
    """The included TracrBench programs, sorted by name."""
    return [p for _, p in sorted(_tracrbench().items())]


def module_from_source(source: str, module_name: str | None = None) -> ModuleType:
    """Execute RASP program source code as a fresh module."""
    if module_name is None:
        module_name = "rasp_src_" + hashlib.sha1(source.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(source, module.__dict__)
    return module


def build_sop(source: str, program: str, max_len: int = MAX_LEN, module_name: str | None = None) -> rasp.SOp:
    """Build the RASP SOp of `program` from (possibly mutated) source code."""
    return get(program).build(module_from_source(source, module_name), max_len)


def reference_source(program: str) -> str:
    return get(program).source_path.read_text()
