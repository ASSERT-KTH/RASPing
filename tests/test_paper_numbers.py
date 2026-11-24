"""The submitted TMLR paper's numbers, recomputed from the frozen v1 result tables.

These tests are the oracle for the cleanup: they pin every number in the submitted
paper (`tmlr-submission-1`) that can be recomputed from `results/v1_tmlr/`, except
Appendix C (the Dyck programs, removed from RaspBugs: docs/FINDINGS.md F35). They are
written as plain pandas on purpose — independent of the analysis code under
`experiments/` — so that a refactor of the analysis code can be checked against them.

Known methodological issues behind some of these numbers are listed in
docs/FINDINGS.md; the tests pin what was *reported*, not what is correct.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V1 = Path(__file__).resolve().parents[1] / "results" / "v1_tmlr"
THRESHOLD = 0.99
PROGRAMS = ["hist", "most_freq", "reverse", "sort"]
FILE_TO_PROGRAM = {"most-freq": "most_freq", "shuffle_dyck1": "shuffle_dyck"}

pytestmark = pytest.mark.skipif(not V1.exists(), reason="results/v1_tmlr not present")


def pct(x) -> float:
    return round(100.0 * float(np.mean(x)), 1)


def round_half_up(x: float) -> int:
    """The paper rounds .5 up (Python's round() rounds half to even)."""
    return int(np.floor(x + 0.5))


@pytest.fixture(scope="module")
def mutants() -> pd.DataFrame:
    return pd.read_csv(V1 / "mutants.csv")


@pytest.fixture(scope="module")
def main_runs(mutants) -> pd.DataFrame:
    runs = pd.read_csv(V1 / "main_repair/runs.csv")
    runs = runs.merge(mutants[["job_id", "mutation_order"]], on="job_id", how="left")
    runs["fixed"] = runs["test_accuracy"] >= THRESHOLD
    return runs


@pytest.fixture(scope="module")
def main847(main_runs) -> pd.DataFrame:
    return main_runs[main_runs["program"].isin(PROGRAMS)]


# --------------------------------------------------------------------------- §4 / App. A


def test_benchmark_size(mutants, main_runs):
    rb = mutants[mutants["in_raspbugs_v1"]]
    assert len(rb) == 1466
    assert (rb["status"] == "BUGGY_MODEL").all()
    assert set(main_runs["job_id"]) == set(rb["job_id"])
    assert rb["program"].value_counts().to_dict() == {
        "shuffle_dyck": 331, "shuffle_dyck2": 288, "most_freq": 284,
        "reverse": 268, "sort": 249, "hist": 46,
    }


def test_section4_descriptive_stats(mutants):
    acc = mutants.loc[mutants["in_raspbugs_v1"], "oracle_accuracy_50"] * 100
    assert round(acc.min(), 2) == 0.00
    assert round(acc.median(), 2) == 2.00
    assert round(acc.mean(), 2) == 36.69
    assert round(acc.max(), 2) == 98.00


TABLE4 = {
    "replace-binary-operator": 1055, "replace-rasp-comparison": 896,
    "replace-comparison-operator": 330, "negate-rasp-sop-select": 288,
    "number-replacer": 283, "negate-rasp-sop-constructor": 160,
    "decrement-integer": 140, "increment-integer": 131, "decrement-rasp-indices": 130,
    "increment-rasp-indices": 127, "negate-rasp-sop-return-stmt": 108,
    "replace-unary-operator": 74, "zero-iteration-for-loop": 19,
    "negate-rasp-sop-aggregate-value": 14, "add-not": 4,
}
CORE_FAMILY = {
    "ReplaceBinaryOperator": "replace-binary-operator",
    "ReplaceComparisonOperator": "replace-comparison-operator",
    "NumberReplacer": "number-replacer",
    "ReplaceUnaryOperator": "replace-unary-operator",
    "ZeroIterationForLoop": "zero-iteration-for-loop",
    "AddNot": "add-not",
}


def test_table4_operator_occurrences(mutants):
    c = Counter()
    for ops in mutants.loc[mutants["in_raspbugs_v1"], "operators"]:
        for op in ops.split(";"):
            if op.startswith("core/"):
                op = CORE_FAMILY[op.removeprefix("core/").split("_")[0]]
            c[op] += 1
    assert dict(c) == TABLE4


# --------------------------------------------------------------------------- Table 1 / 5


TABLE1 = {"hist": (46, 19), "most_freq": (284, 0), "reverse": (268, 128), "sort": (249, 162)}


def test_table1(main847):
    assert len(main847) == 847
    assert main847["fixed"].sum() == 309
    assert pct(main847["fixed"]) == 36.5
    for prog, (n, fixed) in TABLE1.items():
        sub = main847[main847["program"] == prog]
        assert (len(sub), int(sub["fixed"].sum())) == (n, fixed), prog


TABLE5 = {  # program -> order -> (fixed, total)
    "hist": {1: (3, 9), 2: (4, 14), 3: (4, 9), 4: (8, 14)},
    "most_freq": {1: (0, 61), 2: (0, 86), 3: (0, 61), 4: (0, 42), 5: (0, 34)},
    "reverse": {1: (19, 35), 2: (67, 115), 3: (20, 61), 4: (19, 37), 5: (3, 20)},
    "sort": {1: (19, 39), 2: (56, 88), 3: (40, 56), 4: (30, 43), 5: (17, 23)},
    "All": {1: (41, 144), 2: (127, 303), 3: (64, 187), 4: (57, 136), 5: (20, 77)},
}


def test_table5(main847):
    for prog, by_order in TABLE5.items():
        sub = main847 if prog == "All" else main847[main847["program"] == prog]
        got = {
            int(o): (int(g["fixed"].sum()), len(g)) for o, g in sub.groupby("mutation_order")
        }
        assert got == by_order, prog


def test_fig3_medians(main847):
    med = main847.groupby("program")["test_accuracy"].median() * 100
    assert round(med["hist"], 2) == 53.02
    assert round(med["most_freq"], 2) == 35.58
    assert round(med["reverse"], 2) == 98.71
    assert round(med["sort"], 2) == 99.80


def test_fig7_medians(main847):
    med = main847.groupby("mutation_order")["test_accuracy"].median() * 100
    assert [round(med[o], 2) for o in range(1, 6)] == [48.99, 95.42, 64.98, 89.01, 46.70]


# --------------------------------------------------------------------------- §5.3 / Fig 4-5


@pytest.fixture(scope="module")
def baselines1000(mutants) -> pd.DataFrame:
    frames = []
    for m in ("gp", "bfs"):
        df = pd.read_csv(V1 / f"symbolic_baselines/{m}_budget1000_runs.csv")
        frames.append(df)
    df = pd.concat(frames)
    df["program"] = df["program_file"].replace(FILE_TO_PROGRAM)
    return df.merge(mutants[["job_id", "mutation_order"]], on="job_id", how="left")


def test_section53_convergence_endpoints(baselines1000, main847):
    """§5.3 'Wall-Clock Time Comparison' endpoints (orders 4 and 5).

    The baseline curves use the max over candidates of *test* accuracy (F17).
    """
    b = baselines1000[baselines1000["program"].isin(PROGRAMS)]
    b = b.assign(fixed=b["max_best_test_acc"] >= THRESHOLD)
    got = {(m, o): pct(g["fixed"]) for (m, o), g in b.groupby(["method", "mutation_order"])}
    assert got[("gp", 4)] == 22.1 and got[("bfs", 4)] == 29.4
    assert got[("gp", 5)] == 15.6 and got[("bfs", 5)] == 16.9
    # GBPR endpoints quoted in the same paragraph equal the Table 5 test-set rates.
    o = main847.groupby("mutation_order")["fixed"].mean() * 100
    assert round(o[4], 1) == 41.9 and round(o[5], 1) == 26.0


def test_fig4_panel_sizes(main847):
    assert main847["mutation_order"].value_counts().sort_index().tolist() == [144, 303, 187, 136, 77]


# --------------------------------------------------------------------------- Table 8 (App. D)


TABLE8 = {  # order -> [N=100, 1k, 5k, 10k, 25k, 40k, GP, BFS]
    1: [0.0, 9.0, 15.3, 16.0, 27.1, 28.5, 77.8, 100.0],
    2: [0.0, 10.9, 20.1, 26.4, 37.0, 41.9, 56.4, 57.4],
    3: [1.1, 5.9, 8.6, 16.6, 23.5, 34.2, 36.4, 28.9],
    4: [0.0, 8.8, 14.0, 18.4, 36.0, 41.9, 16.9, 20.6],
    5: [0.0, 3.9, 3.9, 14.3, 19.5, 26.0, 16.9, 10.4],
    "All": [0.2, 8.5, 14.3, 20.1, 30.6, 36.5, 45.7, 48.2],
}
N_VALUES = [100, 1000, 5000, 10000, 25000]


def test_table8_spec_size(mutants, main847):
    ref = set(main847["job_id"])
    order = mutants.set_index("job_id")["mutation_order"]
    spec = pd.read_csv(V1 / "spec_size/runs.csv")
    spec = spec[spec["job_id"].isin(ref)]
    spec = spec.assign(mutation_order=spec["job_id"].map(order),
                       fixed=spec["test_accuracy"] >= THRESHOLD)
    cols = {n: spec[spec["n_train"] == n] for n in N_VALUES}
    cols[40000] = main847  # the N=40,000 column reuses the main runs (F8)

    base = {}
    for m in ("gp", "bfs"):
        b = pd.read_csv(V1 / f"spec_size/{m}_budget500_runs.csv")
        b = b[b["job_id"].isin(ref)]
        base[m] = b.assign(mutation_order=b["job_id"].map(order),
                           fixed=b["max_best_test_acc"] >= THRESHOLD)

    for o, expected in TABLE8.items():
        row = []
        for df in [*cols.values(), base["gp"], base["bfs"]]:
            sub = df if o == "All" else df[df["mutation_order"] == o]
            row.append(pct(sub["fixed"]))
        assert row == expected, o


# --------------------------------------------------------------------------- Table 2 / App. E


TABLE2 = {  # program -> buggy acc, random acc, buggy rep., random rep., buggy ep., random ep.
    "hist": (56.3, 100.0, 45.0, 100.0, 385, 231),
    "most_freq": (36.0, 81.2, 0.0, 0.0, 655, 760),
    "reverse": (81.6, 100.0, 40.0, 100.0, 459, 71),
    "sort": (91.2, 100.0, 60.0, 100.0, 429, 102),
    "All": (66.3, 95.3, 36.2, 75.0, 482, 291),
}


@pytest.fixture(scope="module")
def paired(main_runs):
    rnd = pd.read_csv(V1 / "init_control/runs.csv")
    per_mutant = rnd.groupby(["program", "job_id"]).agg(
        random_acc=("test_accuracy", "mean"), random_epochs=("n_epochs", "mean")
    ).reset_index()
    buggy = main_runs.set_index("job_id")
    per_mutant["buggy_acc"] = per_mutant["job_id"].map(buggy["test_accuracy"])
    per_mutant["buggy_epochs"] = per_mutant["job_id"].map(buggy["n_epochs"])
    return rnd, per_mutant


def test_table2_paired_init(paired):
    rnd, pm = paired
    assert len(pm) == 80 and len(rnd) == 400
    for prog, exp in TABLE2.items():
        r = rnd if prog == "All" else rnd[rnd["program"] == prog]
        p = pm if prog == "All" else pm[pm["program"] == prog]
        got = (
            pct(p["buggy_acc"]),
            pct(p["random_acc"]),
            pct(p["buggy_acc"] >= THRESHOLD),
            pct(r["test_accuracy"] >= THRESHOLD),  # fraction of the individual runs
            round_half_up(p["buggy_epochs"].mean()),
            round_half_up(p["random_epochs"].mean()),
        )
        assert got == exp, prog


def test_fig9_outcomes(paired):
    _, pm = paired
    diff = pm["random_acc"] - pm["buggy_acc"]
    wins, losses = int((diff > 0.01).sum()), int((diff < -0.01).sum())
    ties = len(pm) - wins - losses
    assert (wins, ties, losses) == (50, 29, 1)
    tie = pm[diff.abs() <= 0.01]
    assert ((tie["buggy_acc"] >= THRESHOLD) & (tie["random_acc"] >= THRESHOLD)).all()


TABLE9 = {  # program -> state -> (zero% lo, zero% hi, std lo, std hi, max|w|)
    "hist": {"compiled": (94.9, 98.5, 5.87, 13.70, 120.00),
             "repaired_from_compiled": (33.0, 65.2, 5.87, 13.67, 120.46),
             "trained_from_xavier": (0.3, 0.7, 0.20, 0.37, 2.22)},
    "most_freq": {"compiled": (98.8, 99.3, 3.24, 10.34, 120.00),
                  "repaired_from_compiled": (59.2, 84.1, 3.23, 10.34, 120.30),
                  "trained_from_xavier": (0.0, 0.2, 0.09, 0.20, 2.52)},
    "reverse": {"compiled": (99.1, 99.6, 1.63, 4.98, 120.00),
                "repaired_from_compiled": (67.2, 75.8, 1.62, 4.98, 120.30),
                "trained_from_xavier": (0.1, 0.1, 0.10, 0.12, 0.79)},
    "sort": {"compiled": (96.2, 99.4, 2.47, 18.11, 120.00),
             "repaired_from_compiled": (56.5, 76.0, 2.47, 18.11, 120.10),
             "trained_from_xavier": (0.1, 0.2, 0.10, 0.17, 0.92)},
}


def test_table9_weight_stats():
    path = V1 / "init_control/weight_stats.csv"
    if not path.exists():
        pytest.skip("weight_stats.csv not frozen yet")
    ws = pd.read_csv(path)
    for prog, states in TABLE9.items():
        for state, exp in states.items():
            s = ws[(ws["program"] == prog) & (ws["state"] == state)]
            got = (
                round(100 * s["zero_frac"].min(), 1), round(100 * s["zero_frac"].max(), 1),
                round(s["std"].min(), 2), round(s["std"].max(), 2), round(s["max_abs"].max(), 2),
            )
            assert got == exp, (prog, state)
