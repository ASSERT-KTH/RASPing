"""TracrBench programs (registry, specifications, inclusion rules) and its mutation pipeline."""

import importlib.util
import json

import numpy as np

from rasping import data, programs
from rasping.config import DATA_DIR, PROGRAMS, RASPBUGS_DIR, TRACRBENCH

RECORDS = programs.tracrbench_records()
INCLUDED = [r for r in RECORDS if r["included"]]


def _build_manifest():
    spec = importlib.util.spec_from_file_location("build_manifest", RASPBUGS_DIR / "build_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_records_follow_the_rules():
    assert len(RECORDS) == 101 and len({r["name"] for r in RECORDS}) == 101
    for r in INCLUDED:
        assert r["reason"] is None and r["test_accuracy"] == 1.0
        assert max(r["const_baseline"], r["identity_baseline"]) < TRACRBENCH.max_trivial_accuracy
    assert all(r["reason"] for r in RECORDS if not r["included"])


def test_registry_and_files():
    ps = programs.tracrbench_programs()
    assert [p.name for p in ps] == sorted(r["name"] for r in INCLUDED)
    assert not {p.name for p in ps} & set(PROGRAMS)
    for p in ps:
        assert p.benchmark == "tracrbench" and p.mlp_exactness == TRACRBENCH.mlp_exactness
        src = p.source_path.read_text()
        assert src.startswith(programs.TRACRBENCH_HEADER) and src.endswith("\n")  # diffs apply
        assert all((DATA_DIR / p.data_file(s)).exists() for s in ("train", "val", "test"))
    assert programs.get("sort").source_path == RASPBUGS_DIR / "source" / "sort.py"


def test_specification_matches_ground_truth():
    p = programs.tracrbench_programs()[0]
    test = data.load(p.name, "test")
    assert len(test) == 5000
    for x, y in test[:200]:
        assert x[0] == y[0] == "BOS" and 2 <= len(x) - 1 <= 10 and len(x) == len(y)
        assert y[1:] == p.target(x[1:])
    np.random.seed(0)
    from rasping.generators import get_generator

    x, y = get_generator(p.name)(list(p.vocab), 10)
    assert y[1:] == p.target(x[1:])


def test_smallest_program_compiles_exactly():
    r = min(INCLUDED, key=lambda r: r["n_params"])
    from rasping.model import compile_program

    model = compile_program(r["name"])
    assert model.n_params == r["n_params"]
    assert model.exact_match(data.load(r["name"], "test")).all()


def test_batched_exact_match_agrees_with_evaluate():
    from rasping.model import compile_program

    model = compile_program("sort")
    test = data.load("sort", "test")[:300]
    assert (model.exact_match(test) == model.evaluate(test)).all()
    wrong = [(x, [y[0], *reversed(y[1:])]) for x, y in test[:50]]
    assert (model.exact_match(wrong) == model.evaluate(wrong)).all()


def test_exact_match_with_a_single_output_value():
    """A mutant that always outputs 0 (Tracr then drops the output axis) is scored, not a crash."""
    from rasping.model import Model

    p = programs.get("binarize")
    src = p.source_path.read_text().replace("1 if x else 0", "0 if x else 0")
    model = Model(programs.build_sop(src, p.name), p.vocab, 10, p.name, mlp_exactness=p.mlp_exactness)
    test = data.load(p.name, "test")[:200]
    assert (model.exact_match(test) == model.evaluate(test)).all()
    assert model.predict([["BOS", 1, 7]]) == [[0, 0, 0]]


def test_oracle_markers():
    bm = _build_manifest()
    out = "REPRESENTABLE:  False\nAccuracy:  0.25\n"
    assert bm.label({"output": out}) == {"status": "BUGGY_MODEL", "accuracy": 0.25,
                                         "oracle_representable": False}
    assert bm.label({"output": "Accuracy:  0.5\n"}) == {"status": "BUGGY_MODEL", "accuracy": 0.5}


def test_selection_caps_per_program_by_order():
    bm = _build_manifest()
    rows = [{"job_id": f"{p}{i:03d}", "program": p, "mutation_order": 1 + i % 5,
             "representable": i % 3 != 0} for p in ("a", "b") for i in range(100)]
    rows += [{"job_id": "c0", "program": "c", "mutation_order": 1, "representable": True}]
    bm.select(rows, "tracrbench")
    for p, n in (("a", 20), ("b", 20), ("c", 1)):
        sel = [r for r in rows if r["program"] == p and r["selected"]]
        assert len(sel) == n and all(r["representable"] for r in sel)
    orders = [r["mutation_order"] for r in rows if r["program"] == "a" and r["selected"]]
    assert sorted(set(orders)) == [1, 2, 3, 4, 5] and min(orders.count(o) for o in range(1, 6)) == 4
    bm.select(rows, "raspbugs")
    assert all(r["selected"] == r["representable"] for r in rows)


def test_programs_json_is_sorted_and_stable():
    names = [r["name"] for r in json.loads(programs.TRACRBENCH_PROGRAMS.read_text())]
    assert names == sorted(names)


def test_empty_diff_leaves_the_source_unchanged():
    # order-2 mutants whose mutations cancel (e.g. - -> % -> -) have a header but no hunks (F38)
    bm = _build_manifest()
    src = "def f(x):\n    return x - 1\n"
    assert bm.apply_diff(src, "--- mutation diff ---\n") == src
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n def f(x):\n-    return x - 1\n+    return x % 1\n"
    assert bm.apply_diff(src, diff) == "def f(x):\n    return x % 1\n"


def test_canonical_source_ignores_names_only():
    a = '''def make_a() -> rasp.SOp:
    """Doc."""
    sel = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named("sel")  # comment
    return rasp.SelectorWidth(sel).named("a")
'''
    b = '''def make_b():
    s = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named("eq")
    return rasp.SelectorWidth(s).named("b")
'''
    assert programs.canonical_source(a) == programs.canonical_source(b)
    assert programs.canonical_source(a) != programs.canonical_source(a.replace("EQ", "LT"))
    assert programs.canonical_source(a) != programs.canonical_source(a.replace("tokens, rasp.tokens", "tokens, rasp.indices"))


def test_rule_4_no_duplicate_programs():
    keys = [programs.canonical_source(p.source_path.read_text()) for p in programs.tracrbench_programs()]
    assert len(set(keys)) == len(keys)
    dups = {r["name"]: r["reason"] for r in RECORDS if (r["reason"] or "").startswith("duplicate")}
    assert dups == {n: "duplicate of count_frequency (same source up to names)"
                    for n in ("count_occurrences", "histogram")}


def test_manifest_matches_labels_and_architectures():
    import collections

    import pandas as pd

    from rasping import tracrbench
    from rasping.config import TRACRBENCH_DIR

    ms = tracrbench.mutants()
    assert len(ms) == len({m.job_id for m in ms}) == 1153
    per_program = collections.Counter(m.program for m in ms)
    assert set(per_program) <= {p.name for p in programs.tracrbench_programs()}
    assert len(per_program) == 58 and max(per_program.values()) == TRACRBENCH.mutants_per_program
    labels = pd.read_csv(TRACRBENCH_DIR / "mutants.csv.gz")
    sel = labels[labels.selected]
    assert set(sel.job_id) == {m.job_id for m in ms}
    assert (sel.status == "BUGGY_MODEL").all() and sel.representable.all()
    assert set(pd.read_csv(TRACRBENCH_DIR / "architectures.csv").job_id) == set(sel.job_id)
