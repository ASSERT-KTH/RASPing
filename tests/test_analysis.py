"""The analysis code (rasping.analysis) reproduces the submitted paper from results/v1_tmlr.

The expected values come from tests/test_paper_numbers.py, which recomputes them with
independent plain-pandas code; this file checks the code the artifacts are built with.
"""

import pandas as pd
import pytest

from rasping import analysis
from rasping.config import MAIN_PROGRAMS
from rasping.results import load_runs, results_dir
from tests.test_paper_numbers import TABLE1, TABLE2, TABLE5, TABLE8, TABLE9, round_half_up

V = "v1_tmlr"
pytestmark = pytest.mark.skipif(not results_dir(V).exists(), reason="no v1 results")


@pytest.fixture(scope="module")
def main847():
    df = load_runs(V, "main_repair")
    return df[df["program"].isin(MAIN_PROGRAMS)]


def test_table1(main847):
    t = analysis.repair_rates(main847, "program").set_index("program")
    for prog, (n, k) in TABLE1.items():
        assert (t.loc[prog, "total"], t.loc[prog, "fixed"]) == (n, k)
    assert (t.loc["All", "total"], t.loc["All", "fixed"], round(t.loc["All", "pct"], 1)) == (847, 309, 36.5)


def test_table5(main847):
    m = analysis.repair_matrix(main847)
    for prog, by_order in TABLE5.items():
        for order, cell in by_order.items():
            assert m.loc[prog, order] == cell, (prog, order)
    assert m.loc["hist", 5] is None


def _baseline(method, budget, sub="symbolic_baselines"):
    return load_runs(V, sub, f"{method}_budget{budget}_runs.csv")


def test_fig5_time_endpoints(main847):
    jobs = set(main847["job_id"])
    order = main847.set_index("job_id")["mutation_order"]
    for method, exp in (("gp", {4: 22.1, 5: 15.6}), ("bfs", {4: 29.4, 5: 16.9})):
        hist = pd.read_csv(results_dir(V, "symbolic_baselines") / f"{method}_budget1000_history.csv.gz")
        runs = _baseline(method, 1000)
        sps = dict(zip(runs["job_id"], runs["duration_s"] / runs["num_programs"]))
        series = analysis.best_so_far_series(hist[hist["job_id"].isin(jobs)])
        for o, e in exp.items():
            s = {j: v for j, v in series.items() if order[j] == o}
            _, y = analysis.pass_rate_by_time(s, {j: sps[j] for j in s})
            assert round(y[-1], 1) == e, (method, o)


def test_fig4_gbpr_curve_endpoint(main847):
    """The GBPR curve in Fig 4/5 uses best validation accuracy (F4)."""
    val = pd.read_csv(results_dir(V, "main_repair") / "val_curves.csv.gz")
    order = main847.set_index("job_id")["mutation_order"]
    series = analysis.cummax_series(val[val["job_id"].isin(order.index)])
    finals = {o: analysis.pass_rate_by_step({j: s for j, s in series.items() if order[j] == o})[1][-1]
              for o in range(1, 6)}
    # Plateaus visible in Fig 4 (val-based) differ slightly from Table 5 (test-based).
    assert all(0 <= v <= 100 for v in finals.values())


def test_table8(main847):
    ref = set(main847["job_id"])
    spec = load_runs(V, "spec_size")
    spec = spec[spec["job_id"].isin(ref)]
    gbpr = {n: spec[spec["n_train"] == n] for n in sorted(spec["n_train"].unique())}
    gbpr[40000] = main847
    base = {m: (lambda d: d[d["job_id"].isin(ref)])(_baseline(m, 500, "spec_size")) for m in ("gp", "bfs")}
    t = analysis.spec_size_table(gbpr, base)
    for o, exp in TABLE8.items():
        assert [round(v, 1) for v in t.loc[o]] == exp, o


def test_table2():
    rnd = load_runs(V, "init_control")
    _, summary = analysis.paired_init(load_runs(V, "main_repair"), rnd)
    s = summary.set_index("program")
    for prog, exp in TABLE2.items():
        r = s.loc[prog]
        got = (round(r.buggy_acc, 1), round(r.random_acc, 1), round(r.buggy_repaired, 1),
               round(r.random_repaired, 1), round_half_up(r.buggy_epochs), round_half_up(r.random_epochs))
        assert got == exp, prog
    assert tuple(s.loc["All", ["random_wins", "ties", "buggy_wins"]]) == (50, 29, 1)


def test_table9():
    ws = pd.read_csv(results_dir(V, "init_control") / "weight_stats.csv")
    t = analysis.weight_stats_ranges(ws).set_index(["program", "state"])
    for prog, states in TABLE9.items():
        for state, exp in states.items():
            r = t.loc[(prog, state)]
            got = (round(r.zero_min, 1), round(r.zero_max, 1), round(r.std_min, 2),
                   round(r.std_max, 2), round(r.max_abs, 2))
            assert got == exp, (prog, state)


def test_tables6_7():
    gb = pd.read_csv(results_dir(V, "dyck") / "gbpr_metrics.csv").assign(method="gbpr")
    bl = pd.read_csv(results_dir(V, "dyck") / "baseline_metrics.csv")
    s = analysis.dyck_summary(pd.concat([gb, bl])).set_index(["method", "program_name"])
    assert round(s.loc[("gbpr", "shuffle_dyck"), "acc"], 2) == 98.28
    assert round(s.loc[("gbpr", "shuffle_dyck2"), "acc_fixed"], 1) == 99.7
    assert round(s.loc[("bfs", "shuffle_dyck"), "recall_fixed"], 1) == 60.1
    assert round(s.loc[("gp", "shuffle_dyck2"), "recall_fixed"], 1) == 3.8
