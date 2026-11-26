"""Computations behind the paper's tables and figures, on tidy result tables.

Kept free of plotting and file I/O so they can be tested against the paper numbers
(`tests/test_analysis.py`). The experiment scripts under `experiments/*/analyze.py`
call these and render LaTeX / PDF artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rasping.config import THRESHOLD

# --------------------------------------------------------------------------- repair rates


def repair_rates(df: pd.DataFrame, by: list[str] | str | None, acc_col: str = "test_accuracy",
                 threshold: float = THRESHOLD, total_label: str = "All") -> pd.DataFrame:
    """Count / percentage of rows with `acc_col >= threshold`, per group plus an "All" row."""
    by = [by] if isinstance(by, str) else (by or [])
    fixed = df[acc_col] >= threshold

    def agg(mask) -> dict:
        n = int(mask.size)
        k = int(mask.sum())
        return {"fixed": k, "total": n, "pct": 100.0 * k / n if n else np.nan}

    rows = []
    if by:
        for key, idx in df.groupby(by).groups.items():
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(by, key)), **agg(fixed.loc[idx])})
    rows.append({**{b: total_label for b in by}, **agg(fixed)})
    return pd.DataFrame(rows)


def repair_matrix(df: pd.DataFrame, rows: str = "program", cols: str = "mutation_order",
                  acc_col: str = "test_accuracy", threshold: float = THRESHOLD) -> pd.DataFrame:
    """Table 5 layout: (fixed, total) per rows x cols, with "All" margins."""
    out = {}
    row_keys = sorted(df[rows].unique()) + ["All"]
    col_keys = sorted(df[cols].unique()) + ["All"]
    for r in row_keys:
        dr = df if r == "All" else df[df[rows] == r]
        out[r] = {}
        for c in col_keys:
            d = dr if c == "All" else dr[dr[cols] == c]
            out[r][c] = (int((d[acc_col] >= threshold).sum()), len(d)) if len(d) else None
    return pd.DataFrame(out).T.loc[row_keys, col_keys]


# --------------------------------------------------------------------------- progress curves
# Port of experiments/gp_baseline/plot_progress.py (v1) — same aggregation semantics.


def best_so_far_series(history: pd.DataFrame, value: str = "best_test_acc",
                       running_max: bool = True) -> dict[str, list[float]]:
    """Dense per-step series from (sparse) baseline histories, per job.

    `running_max=True` is the v1 curve semantics (F17); with False the value at each
    step is that of the program selected at that step.
    """
    out = {}
    for job_id, h in history.groupby("job_id", sort=False):
        h = h[(h["step"] > 0) & h[value].notna()].sort_values("step", kind="stable")
        if h.empty:
            continue
        steps, vals = h["step"].to_numpy(int), h[value].to_numpy(float)
        dense = np.zeros(steps[-1])
        best, ptr = 0.0, 0
        for s in range(1, steps[-1] + 1):
            while ptr < len(steps) and steps[ptr] == s:
                best = max(best, vals[ptr]) if running_max else vals[ptr]
                ptr += 1
            dense[s - 1] = best
        out[job_id] = dense.tolist()
    return out


def cummax_series(curves: pd.DataFrame, value: str = "val_acc", index: str = "val_eval") -> dict[str, list[float]]:
    """Per-job cumulative maximum of a per-evaluation curve (e.g. GBPR validation accuracy)."""
    return {j: np.maximum.accumulate(c.sort_values(index)[value].to_numpy(float)).tolist()
            for j, c in curves.groupby("job_id", sort=False)}


def pass_rate_by_step(series: dict[str, list[float]], threshold: float = THRESHOLD):
    """% of jobs whose best-so-far value is >= threshold after each step (held at the end)."""
    if not series:
        return np.array([]), np.array([])
    vals = list(series.values())
    n = max(len(s) for s in vals)
    mat = np.array([s + [s[-1]] * (n - len(s)) for s in vals])
    return np.arange(1, n + 1), 100.0 * (mat >= threshold).mean(axis=0)


def pass_rate_by_time(series: dict[str, list[float]], sec_per_step: dict[str, float],
                      threshold: float = THRESHOLD, n_grid: int = 500):
    """As `pass_rate_by_step` on a common wall-clock grid (constant seconds per step per job)."""
    if not series:
        return np.array([]), np.array([])
    max_t = max(len(s) * sec_per_step.get(j, 1.0) for j, s in series.items())
    grid = np.linspace(0, max_t, n_grid)
    out = np.zeros(n_grid)
    for k, t in enumerate(grid):
        v = []
        for j, s in series.items():
            sps = sec_per_step.get(j, 1.0)
            idx = min(int(t / sps), len(s) - 1) if sps > 0 else len(s) - 1
            v.append(s[max(idx, 0)])
        out[k] = 100.0 * np.mean(np.asarray(v) >= threshold)
    return grid, out


def pass_rate_on_time_grid(series: dict[str, tuple], threshold: float = THRESHOLD,
                           n_grid: int = 500):
    """Pass rate over wall-clock time from measured `(times, values)` per job.

    A job's value at time t is its last recorded value at or before t (0 before the
    first record) and is held after its last record.
    """
    if not series:
        return np.array([]), np.array([])
    grid = np.linspace(0, max(float(t[-1]) for t, _ in series.values()), n_grid)
    mat = np.zeros((len(series), n_grid))
    for i, (t, v) in enumerate(series.values()):
        idx = np.searchsorted(np.asarray(t, float), grid, side="right") - 1
        vals = np.asarray(v, float)
        mat[i] = np.where(idx >= 0, vals[np.clip(idx, 0, None)], 0.0)
    return grid, 100.0 * (mat >= threshold).mean(axis=0)


# --------------------------------------------------------------------------- spec size


def spec_size_table(gbpr: dict[int, pd.DataFrame], baselines: dict[str, pd.DataFrame],
                    gbpr_col: str = "test_accuracy", base_col: str = "max_best_test_acc",
                    threshold: float = THRESHOLD) -> pd.DataFrame:
    """Table 8: % repaired by mutation order (rows) for each N and each baseline (columns)."""
    cols = {f"N={n}": (d, gbpr_col) for n, d in sorted(gbpr.items())}
    cols.update({m.upper(): (d, base_col) for m, d in baselines.items()})
    orders = sorted(set().union(*(set(d["mutation_order"]) for d, _ in cols.values())))
    table = {}
    for name, (d, col) in cols.items():
        table[name] = {o: 100.0 * float((d.loc[d["mutation_order"] == o, col] >= threshold).mean())
                       for o in orders}
        table[name]["All"] = 100.0 * float((d[col] >= threshold).mean())
    return pd.DataFrame(table)


# --------------------------------------------------------------------------- init control


def paired_init(buggy: pd.DataFrame, random: pd.DataFrame, margin: float = 0.01,
                threshold: float = THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-mutant pairing (buggy-init run vs mean over random-init seeds) and Table 2 summary."""
    pm = random.groupby(["program", "job_id"]).agg(
        random_acc=("test_accuracy", "mean"), random_epochs=("n_epochs", "mean"),
        random_n_seeds=("test_accuracy", "size"),
    ).reset_index()
    b = buggy.set_index("job_id")
    pm["buggy_acc"] = pm["job_id"].map(b["test_accuracy"])
    pm["buggy_epochs"] = pm["job_id"].map(b["n_epochs"])
    diff = pm["random_acc"] - pm["buggy_acc"]
    pm["outcome"] = np.select([diff > margin, diff < -margin], ["random_wins", "buggy_wins"], "tie")

    rows = []
    for prog in sorted(pm["program"].unique()) + ["All"]:
        p = pm if prog == "All" else pm[pm["program"] == prog]
        r = random if prog == "All" else random[random["program"] == prog]
        rows.append({
            "program": prog, "n": len(p),
            "buggy_acc": 100 * p["buggy_acc"].mean(), "random_acc": 100 * p["random_acc"].mean(),
            "buggy_repaired": 100 * (p["buggy_acc"] >= threshold).mean(),
            # fraction of the individual random-init runs (not of mutants)
            "random_repaired": 100 * (r["test_accuracy"] >= threshold).mean(),
            "buggy_epochs": p["buggy_epochs"].mean(), "random_epochs": p["random_epochs"].mean(),
            "random_wins": int((p["outcome"] == "random_wins").sum()),
            "ties": int((p["outcome"] == "tie").sum()),
            "buggy_wins": int((p["outcome"] == "buggy_wins").sum()),
        })
    return pm, pd.DataFrame(rows)


def weight_stats_ranges(ws: pd.DataFrame) -> pd.DataFrame:
    """Table 9: per program and state, ranges of zero fraction / std and max |w|."""
    g = ws.groupby(["program", "state"])
    return pd.DataFrame({
        "zero_min": 100 * g["zero_frac"].min(), "zero_max": 100 * g["zero_frac"].max(),
        "std_min": g["std"].min(), "std_max": g["std"].max(), "max_abs": g["max_abs"].max(),
    }).reset_index()


# --------------------------------------------------------------------------- init pilot


def init_arm(init: str) -> str:
    """Arm name of an init spec: "from:<job_id>" -> "from", others unchanged."""
    return init.split(":", 1)[0] if init.startswith("from:") else init


def arm_table(runs: pd.DataFrame, arms: list[str], programs: list[str],
              threshold: float = THRESHOLD) -> pd.DataFrame:
    """Per arm: runs, mean start agreement / accuracy, % repaired (all and per program),
    mean final test accuracy and median epochs. `runs` needs columns arm, program,
    test_accuracy, start_agreement, start_test_accuracy, n_epochs."""
    rows = []
    for arm in arms:
        d = runs[runs["arm"] == arm]
        if d.empty:
            continue
        row = {"arm": arm, "runs": len(d),
               "start_agreement": 100 * d["start_agreement"].mean(),
               # runs whose starting weights behave like the compiled mutant (>=99% agreement)
               "behaviour_kept": 100 * (d["start_agreement"] >= threshold).mean(),
               "start_acc": 100 * d["start_test_accuracy"].mean(),
               "repaired": 100 * (d["test_accuracy"] >= threshold).mean()}
        for p in programs:
            dp = d[d["program"] == p]
            row[f"repaired_{p}"] = 100 * (dp["test_accuracy"] >= threshold).mean() if len(dp) else np.nan
        row["test_acc"] = 100 * d["test_accuracy"].mean()
        row["median_epochs"] = d["n_epochs"].median()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_outcomes(runs: pd.DataFrame, arm: str, ref: str, threshold: float = THRESHOLD) -> dict:
    """Repair outcomes of `arm` vs `ref` on the mutants (and seeds) both have."""
    key = ["job_id", "seed"]
    a = runs[runs["arm"] == arm].set_index(key)["test_accuracy"] >= threshold
    b = runs[runs["arm"] == ref].set_index(key)["test_accuracy"] >= threshold
    a, b = a.align(b, join="inner")
    return {"arm": arm, "ref": ref, "pairs": len(a), "only_arm": int((a & ~b).sum()),
            "both": int((a & b).sum()), "only_ref": int((~a & b).sum()), "neither": int((~a & ~b).sum())}



def patience_table(runs: pd.DataFrame, low: int, high: int, programs: list[str],
                   threshold: float = THRESHOLD) -> pd.DataFrame:
    """Per arm: % repaired with early-stopping patience `low` vs `high` on the same mutants
    and seeds (all and per program), median epochs, and how many runs only one repairs.
    `runs` needs columns arm, job_id, seed, program, cfg_early_stopping_patience,
    test_accuracy, n_epochs."""
    key = ["arm", "job_id", "seed"]
    cols = ["program", "test_accuracy", "n_epochs"]
    lo = runs[runs["cfg_early_stopping_patience"] == low].set_index(key)[cols]
    hi = runs[runs["cfg_early_stopping_patience"] == high].set_index(key)[cols]
    m = lo.join(hi, how="inner", lsuffix="_lo", rsuffix="_hi")
    m["rep_lo"] = m["test_accuracy_lo"] >= threshold
    m["rep_hi"] = m["test_accuracy_hi"] >= threshold
    rows = []
    for arm, d in m.groupby(level="arm", sort=False):
        row = {"arm": arm, "pairs": len(d), "repaired_lo": 100 * d["rep_lo"].mean(),
               "repaired_hi": 100 * d["rep_hi"].mean()}
        for p in programs:
            dp = d[d["program_lo"] == p]
            row[f"{p}_lo"] = 100 * dp["rep_lo"].mean() if len(dp) else np.nan
            row[f"{p}_hi"] = 100 * dp["rep_hi"].mean() if len(dp) else np.nan
        row.update(only_hi=int((d["rep_hi"] & ~d["rep_lo"]).sum()),
                   only_lo=int((d["rep_lo"] & ~d["rep_hi"]).sum()),
                   epochs_lo=d["n_epochs_lo"].median(), epochs_hi=d["n_epochs_hi"].median())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- v2: pass@1 over seeds
# v2 reports each method by pass@1: the mean over seeds of "this run repaired the mutant",
# averaged over mutants, with a bootstrap confidence interval that resamples mutants (each
# keeps all its seeds). pass@n (repaired in at least one of n seeds) and the share of
# mutants repaired in some but not all seeds measure seed sensitivity (F39).


def per_mutant(runs: pd.DataFrame, method_col: str = "arm", acc_col: str = "test_accuracy",
               threshold: float = THRESHOLD) -> pd.DataFrame:
    """One row per (method, mutant): program, mutation_order, `runs` (seeds), `repaired`
    (runs at or above the threshold) and `pass1` = repaired / runs."""
    d = runs.assign(_rep=runs[acc_col] >= threshold)
    out = (d.groupby([method_col, "job_id"], sort=False)
           .agg(program=("program", "first"), mutation_order=("mutation_order", "first"),
                runs=("_rep", "size"), repaired=("_rep", "sum"))
           .reset_index().rename(columns={method_col: "method"}))
    out["pass1"] = out["repaired"] / out["runs"]
    return out


def bootstrap_means(values, n_boot: int = 10_000, seed: int = 0, chunk: int = 1000) -> np.ndarray:
    """Means of `n_boot` resamples (with replacement) of `values`."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        out.append(v[rng.integers(0, len(v), (b, len(v)))].mean(axis=1))
    return np.concatenate(out)


def mean_ci(values, level: float = 0.95, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lower, upper): percentile bootstrap interval of the mean."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(bootstrap_means(v, n_boot, seed), [(1 - level) / 2, (1 + level) / 2])
    return float(v.mean()), float(lo), float(hi)


def pass_at_1(pm: pd.DataFrame, by: str | None = None, methods: list[str] | None = None,
              total_label: str = "All", **ci) -> pd.DataFrame:
    """Per method (and group of `by`, plus a `total_label` row): mutants, pass@1 with its
    bootstrap CI and pass@n, all in %. `pm` comes from `per_mutant`."""
    methods = methods or list(dict.fromkeys(pm["method"]))
    rows = []
    for m in methods:
        d = pm[pm["method"] == m]
        groups = ([(k, g) for k, g in d.groupby(by)] if by else []) + [(total_label, d)]
        for key, g in groups:
            p, lo, hi = mean_ci(g["pass1"], **ci)
            rows.append({"method": m, **({by: key} if by else {}), "mutants": len(g),
                         "seeds": int(g["runs"].max()) if len(g) else 0,
                         "pass1": 100 * p, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                         "pass_n": 100 * (g["repaired"] > 0).mean() if len(g) else np.nan})
    return pd.DataFrame(rows)


def seed_agreement(pm: pd.DataFrame, methods: list[str] | None = None) -> pd.DataFrame:
    """Per method: % of mutants repaired in no seed, in some but not all, and in all seeds."""
    methods = methods or list(dict.fromkeys(pm["method"]))
    rows = []
    for m in methods:
        d = pm[pm["method"] == m]
        rows.append({"method": m, "mutants": len(d), "seeds": int(d["runs"].max()),
                     "none": 100 * (d["repaired"] == 0).mean(),
                     "some": 100 * ((d["repaired"] > 0) & (d["repaired"] < d["runs"])).mean(),
                     "all": 100 * (d["repaired"] == d["runs"]).mean(),
                     "pass1": 100 * d["pass1"].mean(), "pass_n": 100 * (d["repaired"] > 0).mean()})
    return pd.DataFrame(rows)


def paired_difference(pm: pd.DataFrame, a: str, b: str, by: str | None = None,
                      total_label: str = "All", **ci) -> pd.DataFrame:
    """pass@1 of method `a` minus `b` on the mutants both have (in points), with a paired
    bootstrap CI; counts of mutants where a is better / worse / equal, and a two-sided sign
    test on those counts."""
    from scipy.stats import binomtest

    pa = pm[pm["method"] == a].set_index("job_id")
    pb = pm[pm["method"] == b].set_index("job_id")
    j = pa[["pass1", "program", "mutation_order"]].join(pb[["pass1"]], rsuffix="_b", how="inner")
    j["diff"] = j["pass1"] - j["pass1_b"]
    groups = ([(k, g) for k, g in j.groupby(by)] if by else []) + [(total_label, j)]
    rows = []
    for key, g in groups:
        d, lo, hi = mean_ci(g["diff"], **ci)
        better, worse = int((g["diff"] > 0).sum()), int((g["diff"] < 0).sum())
        p = binomtest(better, better + worse).pvalue if better + worse else 1.0
        rows.append({"a": a, "b": b, **({by: key} if by else {}), "mutants": len(g),
                     "diff": 100 * d, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                     "a_better": better, "b_better": worse, "equal": len(g) - better - worse,
                     "sign_p": float(p)})
    return pd.DataFrame(rows)


def order_trend(pm: pd.DataFrame, methods: list[str] | None = None, n_boot: int = 2000,
                seed: int = 0) -> pd.DataFrame:
    """Change in pass@1 per mutation order (points), per method: least-squares slope of
    per-mutant pass@1 on mutation order with one intercept per program, so the program mix
    of each order doesn't masquerade as an order effect. CI: bootstrap over mutants."""
    methods = methods or list(dict.fromkeys(pm["method"]))
    rng = np.random.default_rng(seed)

    def slope(X, y):  # programs missing from a resample give zero columns: min-norm solution
        return np.linalg.lstsq(X, y, rcond=None)[0][0]

    rows = []
    for m in methods:
        d = pm[pm["method"] == m].reset_index(drop=True)
        X = np.column_stack([d["mutation_order"].to_numpy(float),
                             pd.get_dummies(d["program"]).to_numpy(float)])
        y = d["pass1"].to_numpy(float)
        s = slope(X, y)
        boots = [slope(X[i], y[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(n_boot))]
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append({"method": m, "mutants": len(d), "slope": 100 * s,
                     "ci_lo": 100 * lo, "ci_hi": 100 * hi})
    return pd.DataFrame(rows)


def baseline_at_budget(runs: pd.DataFrame, history: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Symbolic-baseline outcome per mutant after `budget` evaluated programs: the test
    accuracy of the program selected at that point (F17), i.e. the last history change
    point at or before `budget`; the record's final value at its own budget. Returns
    job_id, seed, test_accuracy."""
    full = runs["budget"].iloc[0] if len(runs) else budget
    if budget >= full:
        return runs[["job_id", "seed", "test_acc"]].rename(columns={"test_acc": "test_accuracy"})
    h = history[history["step"] <= budget].sort_values("step").groupby("job_id")["best_test_acc"].last()
    out = runs[["job_id", "seed"]].copy()
    out["test_accuracy"] = out["job_id"].map(h).fillna(0.0).to_numpy()
    return out
