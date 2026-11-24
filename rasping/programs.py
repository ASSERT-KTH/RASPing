"""Registry of the base RASP programs.

Each program has a canonical name (`most_freq`, `shuffle_dyck`, ...), the file prefix
of its specification in `data/` (historically `most-freq`, `shuffle_dyck1`), an input
vocabulary, and a factory that builds the RASP SOp from a module containing the
program's source (the reference source in `raspbugs/source/` or a mutant's source).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

from tracr.rasp import rasp

from rasping.config import MAX_LEN, RASPBUGS_DIR

SOURCE_DIR = RASPBUGS_DIR / "source"


@dataclass(frozen=True)
class Program:
    name: str
    data_name: str
    vocab: tuple
    _factory: Callable[[ModuleType, int, tuple], rasp.SOp]

    def build(self, module: ModuleType, max_len: int = MAX_LEN) -> rasp.SOp:
        """Build the SOp from a module that defines this program's `make_*` function."""
        return self._factory(module, max_len, self.vocab)

    @property
    def source_path(self) -> Path:
        return SOURCE_DIR / f"{self.name}.py"


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
    "shuffle_dyck": Program(
        "shuffle_dyck", "shuffle_dyck1", ("(", ")"), lambda m, n, v: m.make_shuffle_dyck(["()"])
    ),
    "shuffle_dyck2": Program(
        "shuffle_dyck2", "shuffle_dyck2", ("(", ")", "{", "}"), lambda m, n, v: m.make_shuffle_dyck2()
    ),
}

#: Historical names still found in data files and old result records.
ALIASES = {p.data_name: p.name for p in _PROGRAMS.values()}


def canonical(name: str) -> str:
    """Map any historical program name (e.g. `most-freq`, `shuffle_dyck1`) to the canonical one."""
    name = ALIASES.get(name, name)
    if name not in _PROGRAMS:
        raise KeyError(f"Unknown program {name!r}; known: {sorted(_PROGRAMS)}")
    return name


def get(name: str) -> Program:
    return _PROGRAMS[canonical(name)]


def all_programs() -> list[Program]:
    return list(_PROGRAMS.values())


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
