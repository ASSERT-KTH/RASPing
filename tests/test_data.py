"""Specification datasets: generation, splitting, persistence and encoding."""

import pytest

from rasping import data
from rasping.config import MAX_LEN, PROGRAMS
from rasping.model import compile_program


@pytest.mark.parametrize("name", PROGRAMS)
def test_generation_shapes(name):
    exhaustive = name.startswith("shuffle_dyck")
    ds = data.generate(name, MAX_LEN, 2000, exhaustive=exhaustive)
    assert len(ds) == 2000
    for x, y in ds:
        assert 2 <= len(x) - 1 <= MAX_LEN
        assert len(x) == len(y)
        assert x[0] == y[0] == "BOS"


def test_remove_duplicates():
    ds = data.generate("reverse", MAX_LEN, 100, remove_duplicates=True)
    assert len({tuple(x) for x, _ in ds}) == len(ds) <= 100


@pytest.mark.parametrize("name", PROGRAMS)
def test_exhaustive_generation(name):
    ds = data.generate(name, 3, size=1000, exhaustive=True)
    assert all(len(x) <= 4 and x[0] == y[0] == "BOS" for x, y in ds)
    assert len(data.generate(name, 3, size=2, exhaustive=True)) <= 2


def test_split_train_val_test():
    d = list(range(100))
    assert [len(s) for s in data.split_train_val_test(d)] == [80, 10, 10]
    assert [len(s) for s in data.split_train_val_test(d, 0.6, 0.2)] == [60, 20, 20]
    assert sorted(sum(data.split_train_val_test(d), [])) == d


@pytest.mark.parametrize("name", PROGRAMS)
def test_save_load_roundtrip(tmp_path, name):
    ds = data.generate(name, MAX_LEN, 10, exhaustive=name.startswith("shuffle_dyck"))
    for split, part in zip(("train", "val", "test"), data.split_train_val_test(ds)):
        path = data.save(name, split, part, data_dir=tmp_path)
        assert path.name.startswith(data.dataset_path(name, split).name.split("_")[0])
        assert data.load(name, split, data_dir=tmp_path) == [(list(x), list(y)) for x, y in part]


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load("reverse", "train", data_dir=tmp_path)
    with pytest.raises(KeyError):
        data.load("nonexistent_program", "train", data_dir=tmp_path)


def test_committed_specification_sizes():
    """The specifications in data/ that all v1 results were computed on."""
    sizes = {p: [len(data.load(p, s)) for s in ("train", "val", "test")] for p in PROGRAMS}
    for p in ("hist", "most_freq", "reverse", "sort", "shuffle_dyck2"):
        assert sizes[p] == [40000, 5000, 5000], p
    assert sizes["shuffle_dyck"] == [1635, 204, 205]


def test_encode_masks_and_pads():
    model = compile_program("reverse")
    ds = [(["BOS", "a", "b"], ["BOS", "b", "a"])]
    X, Y = data.encode(ds, model)
    assert X.shape == Y.shape == (1, MAX_LEN + 1)
    assert int(X[0, -1]) == model.pad_token
    assert float(model.accuracy(X, Y)) == 1.0
