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


# --------------------------------------------------------------------------- dyck


def dyck_summary(metrics: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Table 6/7 columns: means and % of mutants >= threshold per program (and method)."""
    if "method" in metrics.columns:
        metrics = metrics.assign(method=metrics["method"].str.lower())
    keys = [k for k in ("method", "program_name") if k in metrics.columns]
    rows = []
    for key, d in metrics.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(keys, key)), "n": len(d),
            "acc": 100 * d["full_seq_exact_match_acc"].mean(),
            "acc_fixed": 100 * (d["full_seq_exact_match_acc"] >= threshold).mean(),
            "recall": 100 * d["positive_seq_exact_match_acc"].mean(),
            "recall_fixed": 100 * (d["positive_seq_exact_match_acc"] >= threshold).mean(),
            "bal_acc": 100 * d["token_balanced_accuracy"].mean(),
            "bal_acc_fixed": 100 * (d["token_balanced_accuracy"] >= threshold).mean(),
            "macro_f1": 100 * d["token_macro_f1"].mean(),
        })
    return pd.DataFrame(rows)
