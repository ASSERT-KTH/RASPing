"""v2 analysis: pass@1 over seeds, bootstrap CIs, paired differences, order trend, budgets."""

import numpy as np
import pandas as pd
import pytest

from rasping import analysis


def runs_table(outcomes: dict[str, dict[str, list[bool]]], program="sort", order=1):
    """outcomes[method][job] = repaired per seed -> a runs table."""
    rows = [{"arm": m, "job_id": j, "seed": s, "program": program, "mutation_order": order,
             "test_accuracy": 1.0 if r else 0.5}
            for m, jobs in outcomes.items() for j, rs in jobs.items() for s, r in enumerate(rs)]
    return pd.DataFrame(rows)


def test_per_mutant_and_pass_at_1():
    runs = runs_table({"compiled": {"a": [1, 0, 0], "b": [1, 1, 1], "c": [0, 0, 0]},
                       "xavier": {"a": [1, 1, 1], "b": [1, 1, 1], "c": [0, 0, 1]}})
    pm = analysis.per_mutant(runs)
    c = pm[pm.method == "compiled"].set_index("job_id")
    assert c.loc["a", "repaired"] == 1 and c.loc["a", "runs"] == 3
    assert c.loc["a", "pass1"] == pytest.approx(1 / 3)
    t = analysis.pass_at_1(pm, n_boot=500).set_index("method")
    assert t.loc["compiled", "pass1"] == pytest.approx(100 * (1 / 3 + 1 + 0) / 3)
    assert t.loc["compiled", "pass_n"] == pytest.approx(200 / 3)
    assert t.loc["xavier", "pass_n"] == pytest.approx(100)
    for _, r in t.iterrows():
        assert r.ci_lo <= r.pass1 <= r.ci_hi and r.seeds == 3 and r.mutants == 3
    s = analysis.seed_agreement(pm).set_index("method")
    assert list(s.loc["compiled", ["none", "some", "all"]]) == pytest.approx([100 / 3] * 3)


def test_pass_at_1_groups_and_degenerate_ci():
    runs = pd.concat([runs_table({"x": {f"h{i}": [1] for i in range(4)}}, program="hist"),
                      runs_table({"x": {f"s{i}": [0] for i in range(4)}}, program="sort")])
    t = analysis.pass_at_1(analysis.per_mutant(runs), by="program", n_boot=500).set_index("program")
    assert list(t.index) == ["hist", "sort", "All"]
    assert (t.loc["hist", ["pass1", "ci_lo", "ci_hi"]] == 100).all()
    assert (t.loc["sort", ["pass1", "ci_lo", "ci_hi"]] == 0).all()
    assert t.loc["All", "pass1"] == 50 and 0 < t.loc["All", "ci_lo"] < 50 < t.loc["All", "ci_hi"] < 100


def test_bootstrap_ci_covers_the_truth():
    rng = np.random.default_rng(1)
    hits = 0
    for i in range(200):
        _, lo, hi = analysis.mean_ci(rng.random(300) < 0.3, n_boot=1000, seed=i)
        hits += lo <= 0.3 <= hi
    assert 180 <= hits <= 200  # nominal 95%


def test_paired_difference():
    jobs = {f"j{i}": [1, 1, 1] for i in range(20)}
    worse = {f"j{i}": [1, 0, 0] if i < 15 else [1, 1, 1] for i in range(20)}
    pm = analysis.per_mutant(runs_table({"a": jobs, "b": worse, "c": {"j0": [1, 1, 1]}}))
    d = analysis.paired_difference(pm, "a", "b", n_boot=500).iloc[0]
    assert d.mutants == 20 and (d.a_better, d.b_better, d.equal) == (15, 0, 5)
    assert d["diff"] == pytest.approx(100 * 15 * (2 / 3) / 20) and d.ci_lo > 0
    assert d.sign_p < 1e-3
    assert analysis.paired_difference(pm, "a", "c").iloc[0].mutants == 1  # only shared mutants


def test_order_trend_separates_order_from_program_mix():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(600):
        prog = "easy" if i % 2 else "hard"
        order = rng.integers(1, 6) if prog == "hard" else rng.integers(1, 3)  # hard ones: higher orders
        base = 0.9 if prog == "easy" else 0.3
        for method, slope in (("flat", 0.0), ("falling", -0.1)):
            p = np.clip(base + slope * (order - 1), 0, 1)
            rows.append({"method": method, "job_id": f"m{i}", "program": prog,
                         "mutation_order": order, "runs": 1, "repaired": int(rng.random() < p)})
    pm = pd.DataFrame(rows).assign(pass1=lambda d: d.repaired / d.runs)
    t = analysis.order_trend(pm, n_boot=300).set_index("method")
    assert t.loc["flat", "ci_lo"] < 0 < t.loc["flat", "ci_hi"]  # no order effect despite the mix
    assert t.loc["falling", "ci_lo"] < -10 * 0.5 and t.loc["falling", "ci_hi"] < 0
    # without per-program intercepts, the flat method would look like it degrades with order
    naive = np.polyfit(pm[pm.method == "flat"].mutation_order, pm[pm.method == "flat"].pass1, 1)[0]
    assert naive < -0.05


def test_baseline_at_budget():
    runs = pd.DataFrame({"budget": [100, 100], "job_id": ["a", "b"], "seed": [42, 42],
                         "test_acc": [1.0, 0.4]})
    hist = pd.DataFrame({"job_id": ["a", "a", "a", "b", "b"], "step": [1, 20, 60, 1, 100],
                         "best_test_acc": [0.0, 0.5, 1.0, 0.1, 0.4]})

    def at(budget):
        t = analysis.baseline_at_budget(runs, hist, budget)
        return dict(zip(t["job_id"], t["test_accuracy"]))

    assert at(10) == {"a": 0.0, "b": 0.1}
    assert at(60) == {"a": 1.0, "b": 0.1}
    assert at(100) == {"a": 1.0, "b": 0.4}
