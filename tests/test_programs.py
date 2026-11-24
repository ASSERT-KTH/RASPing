"""Program registry and compiled reference programs."""

import numpy as np
import pytest

from rasping import data, programs
from rasping.config import MAX_LEN, PROGRAMS
from rasping.model import compile_program


def test_aliases():
    assert programs.canonical("most-freq") == "most_freq"
    assert programs.canonical("shuffle_dyck1") == "shuffle_dyck"
    assert programs.canonical("sort") == "sort"
    with pytest.raises(KeyError):
        programs.canonical("nope")


def test_registry_matches_config():
    assert {p.name for p in programs.all_programs()} == set(PROGRAMS)
    assert all(p.source_path.exists() for p in programs.all_programs())


@pytest.mark.parametrize("name", PROGRAMS)
def test_reference_program_is_correct(name):
    """The compiled reference program matches the ground-truth generator."""
    np.random.seed(0)
    ds = data.generate(name, MAX_LEN, 100, exhaustive=name.startswith("shuffle_dyck"))
    model = compile_program(name)
    assert model.name == name and model.max_len == MAX_LEN
    assert model.evaluate(ds).all()
    X, Y = data.encode(ds, model)
    assert float(model.accuracy(X, Y)) == 1.0


@pytest.mark.parametrize("name", PROGRAMS)
def test_reference_program_on_committed_test_split(name):
    """The committed specification agrees with the reference program."""
    ds = data.load(name, "test")[:300]
    model = compile_program(name)
    X, Y = data.encode(ds, model)
    assert float(model.accuracy(X, Y)) == 1.0
