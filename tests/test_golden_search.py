"""Regression test: the GP and BFS baselines reproduce the v1 (TMLR) search code exactly.

`tests/golden/golden_v1.json["search"]` was recorded with `experiments/gp_baseline/`
at tag `tmlr-submission-1`: for one hist and one sort mutant, on a tiny specification
(the first 60/20/40 lines of the train/val/test files),

    run_gp_for_bug(entry, tmp, vocab, budget=12, population_size=4, tournament_k=2,
                   seed=7, log_every=100, eval_timeout=60)
    run_exhaustive_for_bug(entry, tmp, vocab, max_depth=1, budget=12, log_every=100,
                           eval_timeout=60)

storing the result dicts without `duration_s` and with `best_source` replaced by its
sha256. The ported functions must return the same values for every recorded key.
"""

import hashlib
import json
from pathlib import Path

import pytest

from rasping import programs, raspbugs
from rasping.baselines import bfs, gp
from rasping.config import DATA_DIR

GOLDEN = json.loads((Path(__file__).parent / "golden" / "golden_v1.json").read_text())["search"]
N_LINES = {"train": 60, "val": 20, "test": 40}


@pytest.fixture(scope="module")
def tiny_data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    for program in GOLDEN:
        name = programs.get(program).data_name
        for split, n in N_LINES.items():
            lines = (DATA_DIR / f"{name}_{split}.jsonl").read_text().splitlines(keepends=True)
            (d / f"{name}_{split}.jsonl").write_text("".join(lines[:n]))
    return d


def run(method: str, mutant, data_dir: Path) -> dict:
    if method == "gp":
        return gp.run_gp_for_mutant(mutant, data_dir, budget=12, population_size=4,
                                    tournament_k=2, seed=7, log_every=100, eval_timeout=60)
    return bfs.run_bfs_for_mutant(mutant, data_dir, max_depth=1, budget=12, log_every=100,
                                  eval_timeout=60)


@pytest.mark.parametrize("method", ["gp", "bfs"])
@pytest.mark.parametrize("program", sorted(GOLDEN))
def test_golden_search(program, method, tiny_data_dir):
    g = GOLDEN[program]
    expected = dict(g[method])
    got = run(method, raspbugs.get(g["job_id"]), tiny_data_dir)
    assert all("time_s" in e for e in got["program_history"])
    got["program_history"] = [{k: v for k, v in e.items() if k != "time_s"}
                              for e in got["program_history"]]  # wall-clock: not reproducible

    assert hashlib.sha256(got["best_source"].encode()).hexdigest() == expected.pop("best_source_sha")
    for key, value in expected.items():
        assert got[key] == value, key
    assert got["method"] == method and got["budget"] == 12 and got["program"] == program
    assert got["duration_s"] > 0
