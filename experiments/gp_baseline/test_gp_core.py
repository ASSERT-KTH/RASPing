import random
import difflib
from pathlib import Path

import parso
import pytest

from experiments.gp_baseline.gp_core import (
    discover_mutation_sites,
    apply_mutation_once,
    run_gp,
)


# Minimal valid program source that our operators can touch
MINI_SORT_SRC = """
from tracr.rasp import rasp

def make_sort(vals=rasp.tokens, keys=rasp.tokens, max_seq_len=5, min_key=1):
    smaller = rasp.Select(keys, keys, rasp.Comparison.GT).named("smaller")
    target_pos = rasp.SelectorWidth(smaller).named("target_pos")
    sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(sel_new, vals).named("sort")
"""


@pytest.fixture
def mini_tree():
    return parso.parse(MINI_SORT_SRC)


def test_discover_mutation_sites_nonempty(mini_tree):
    sites = discover_mutation_sites(mini_tree)
    assert isinstance(sites, list)
    assert len(sites) == 14, "Expected 14 mutation sites in minimal program"


def test_apply_mutation_once_changes_source():
    rng = random.Random()
    mutated = apply_mutation_once(MINI_SORT_SRC, rng)
    assert mutated is not None
    assert mutated != MINI_SORT_SRC, "Mutation should change the source code"


def test_run_gp_respects_attempt_budget(tmp_path: Path):
    # Create tiny synthetic dataset on disk
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Program name used by run_gp is canonicalized externally; we call run_gp directly
    # Build tiny reverse-like dataset but compatible with sort interface
    # Inputs and outputs include BOS at index 0; we ignore BOS in fitness
    import json

    samples = [
        {"input": ["BOS", 2, 1], "output": ["BOS", 1, 2]},
        {"input": ["BOS", 3, 2], "output": ["BOS", 2, 3]},
    ]
    (data_dir / "sort_train.jsonl").write_text("\n".join(json.dumps(x) for x in samples))
    (data_dir / "sort_val.jsonl").write_text("\n".join(json.dumps(x) for x in samples))
    (data_dir / "sort_test.jsonl").write_text("\n".join(json.dumps(x) for x in samples))

    # Attempt budget small to finish fast
    result = run_gp(
        base_source=MINI_SORT_SRC,
        program_name_raw="sort",
        data_dir=data_dir,
        accepted_inputs=[1, 2, 3],
        budget=100,
        population_size=16,
        tournament_k=2,
        log_every=0,
    )

    # Budget counted by total programs generated
    assert result["num_programs"] <= 100
    # print("\n".join(difflib.unified_diff(MINI_SORT_SRC.splitlines(), result["best_source"].splitlines())))
    # Should return metrics and a source
    assert result["train_acc"] == 1.0
    assert result["test_acc"] == 1.0
    assert "best_source" in result and isinstance(result["best_source"], str)
