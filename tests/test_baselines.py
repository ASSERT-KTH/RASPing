"""Unit tests of the symbolic baselines (port of experiments/gp_baseline/test_gp_core.py)."""

import collections
import json
import random
from pathlib import Path

import parso
import pytest

from rasping.baselines import bfs, gp, records
from rasping.baselines.common import (
    all_single_step_neighbors,
    apply_mutation_once,
    compute_max_length,
    discover_mutation_sites,
)

# Minimal valid program source that the operators can touch
MINI_SORT_SRC = """
from tracr.rasp import rasp

def make_sort(vals=rasp.tokens, keys=rasp.tokens, max_seq_len=5, min_key=1):
    smaller = rasp.Select(keys, keys, rasp.Comparison.GT).named("smaller")
    target_pos = rasp.SelectorWidth(smaller).named("target_pos")
    sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(sel_new, vals).named("sort")
"""

# One site per (operator, matching node). The v1 test asserted 14 sites in total, which
# was wrong (docs/FINDINGS.md F13): there are 16.
EXPECTED_SITES = {
    "replace-rasp-comparison": 2,  # Comparison.GT, Comparison.EQ
    "increment-integer": 2,  # the literals 5 and 1
    "decrement-integer": 2,
    "increment-rasp-indices": 1,
    "decrement-rasp-indices": 1,
    "negate-rasp-sop-return-stmt": 1,
    "negate-rasp-sop-select": 2,
    "negate-rasp-sop-aggregate-value": 1,
    "negate-rasp-sop-constructor": 2,  # SelectorWidth, Aggregate
    "NumberReplacer": 2,  # the literals 5 and 1
}


@pytest.fixture
def mini_tree():
    return parso.parse(MINI_SORT_SRC)


@pytest.fixture
def sort_data_dir(tmp_path: Path) -> Path:
    """Tiny sort specification (BOS at index 0 of inputs and outputs)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples = [
        {"input": ["BOS", 2, 1], "output": ["BOS", 1, 2]},
        {"input": ["BOS", 3, 2], "output": ["BOS", 2, 3]},
    ]
    for split in ("train", "val", "test"):
        (data_dir / f"sort_{split}.jsonl").write_text("\n".join(json.dumps(x) for x in samples))
    return data_dir


def test_discover_mutation_sites(mini_tree):
    sites = discover_mutation_sites(mini_tree)
    assert isinstance(sites, list)
    assert dict(collections.Counter(s.operator_name for s in sites)) == EXPECTED_SITES
    assert len(sites) == sum(EXPECTED_SITES.values()) == 16


@pytest.mark.parametrize("seed", range(5))
def test_apply_mutation_once_changes_source(seed):
    mutated = apply_mutation_once(MINI_SORT_SRC, random.Random(seed))
    assert mutated is not None
    assert mutated != MINI_SORT_SRC, "Mutation should change the source code"


def test_all_single_step_neighbors():
    neighbors = all_single_step_neighbors(MINI_SORT_SRC)
    assert neighbors and MINI_SORT_SRC not in neighbors
    # at most one neighbour per (site, position)
    n_positions = sum(len(s.indices) for s in discover_mutation_sites(parso.parse(MINI_SORT_SRC)))
    assert len(neighbors) <= n_positions


def test_compute_max_length_counts_bos(sort_data_dir):
    assert compute_max_length("sort", sort_data_dir) == 3


def test_run_gp_respects_attempt_budget(sort_data_dir):
    result = gp.run_gp(
        base_source=MINI_SORT_SRC,
        program="sort",
        data_dir=sort_data_dir,
        budget=100,
        population_size=16,
        tournament_k=2,
        log_every=0,
    )
    # Budget counted by total programs generated
    assert result["num_programs"] <= 100
    assert len(result["program_history"]) == result["num_programs"]
    assert result["train_acc"] == 1.0
    assert result["test_acc"] == 1.0
    assert isinstance(result["best_source"], str)


def test_run_bfs_respects_attempt_budget(sort_data_dir):
    result = bfs.run_bfs(MINI_SORT_SRC, "sort", sort_data_dir, max_depth=2, budget=20, log_every=0)
    assert result["num_programs"] <= 20
    assert [e["step"] for e in result["program_history"]] == list(range(1, result["num_programs"] + 1))
    assert result["status"] in ("complete", "success_early")
    assert (result["status"] == "success_early") == (result["train_acc"] >= 1.0)


def test_records_tables():
    hist = [
        {"step": 1, "generation": 0, "status": "ok", "train_acc": 0.1, "test_acc": 0.0,
         "best_train_acc": 0.1, "best_test_acc": 0.0},
        {"step": 2, "generation": 1, "status": "ok", "train_acc": 0.0, "test_acc": 0.0,
         "best_train_acc": 0.1, "best_test_acc": 0.0},
        {"step": 3, "generation": 1, "status": "ok", "train_acc": 0.5, "test_acc": 0.4,
         "best_train_acc": 0.5, "best_test_acc": 0.4},
        {"step": 4, "generation": 3, "status": "duplicate", "train_acc": 0.5, "test_acc": 0.4,
         "best_train_acc": 0.5, "best_test_acc": 0.4},
        {"step": 5, "generation": 3, "status": "ok", "train_acc": 0.0, "test_acc": 0.0,
         "best_train_acc": 0.5, "best_test_acc": 0.4},
    ]
    rec = {"job_id": "x", "program_name": "most_freq", "status": "complete", "num_programs": 5,
           "train_acc": 0.5, "test_acc": 0.4, "seed": 42, "program_history": hist}
    runs, h = records.tables([rec], "gp", 5)
    assert runs.iloc[0][["program_file", "history_len", "max_step", "max_best_test_acc"]].tolist() \
        == ["most_freq", 5, 5, 0.4]
    assert h["step"].tolist() == [1, 3, 5]  # change points + last entry
