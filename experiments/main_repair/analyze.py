"""Main repair results (paper §5.2, App. B).

    python experiments/main_repair/analyze.py --version v1_tmlr
    python experiments/main_repair/analyze.py --version v2

Reads results/<version>/main_repair/runs.csv; writes paper_artifacts/<version>/.

v1 (one compiled-start run per mutant): Table 1, Table 5, Figure 3 and Figure 7.
"Before" accuracies: v2 runs record the compiled mutant's accuracy on the full test
split; v1 only has the mutation oracle's accuracy on 50 test samples (F7).

v2 (several starts x seeds per mutant; `config.V2_INITS`, `V2_SEEDS`): every method is
reported by pass@1, the mean over seeds of "this run repaired the mutant", averaged over
mutants, with a 95% bootstrap CI over mutants (`analysis.pass_at_1`). GP and BFS (one
run per mutant, `results/<version>/symbolic_baselines/`) join the tables when present,
at `config.BASELINE_BUDGET`.
    tables/v2_pass1_by_program.tex   pass@1 [CI] per method x program
    tables/v2_pass1_by_order.tex     pass@1 [CI] per method x mutation order
    tables/v2_seed_agreement.tex     GBPR starts: mutants repaired in no / some / all seeds
    tables/v2_paired.tex             pass@1 differences between methods on the same mutants
    tables/v2_order_trend.tex        change in pass@1 per mutation order, per method
    tables/v2_baseline_budgets.tex   GP/BFS pass@1 at each budget in `config.BASELINE_BUDGETS`
    figures/v2_pass1_by_order.pdf    pass@1 per mutation order with 95% CIs
"""

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rasping import analysis, raspbugs
from rasping.baselines import EXPERIMENT as BASELINES
from rasping.baselines import records
from rasping.config import (
    BASELINE_BUDGET,
    BASELINE_BUDGETS,
    METHOD_LABEL,
    PROGRAMS,
    V2_INITS,
)
from rasping.results import load_runs, mutant_table, results_dir
from rasping.tables import latex_table, pct, save_figure, tt, write_table


def before_accuracy(df):
    if "initial_test_accuracy" in df.columns:
        return df["initial_test_accuracy"]
    oracle = {m.job_id: m.oracle_accuracy_50 for m in raspbugs.mutants()}
    return df["job_id"].map(oracle)


def table1(df, version):
    t = analysis.repair_rates(df, "program")
    rows = [[tt(r.program) if r.program != "All" else r"\textbf{All}", r.total, r.fixed, pct(r.pct)]
            for r in t.itertuples()]
    write_table(version, "table1_repair_by_program.tex", latex_table(
        ["Program", r"\# Mutants", r"\# Fixed ($\geq$99\%)", r"\% Fixed"], rows,
        midrule_before_last=True))


def table5(df, version):
    m = analysis.repair_matrix(df)
    rows = []
    for prog, r in m.iterrows():
        cells = ["--" if c is None else f"{c[0]}/{c[1]} ({100 * c[0] / c[1]:.1f}\\%)" for c in r]
        rows.append([tt(prog) if prog != "All" else r"\textbf{All}", *cells])
    header = ["Program", *[f"Order {c}" for c in m.columns[:-1]], "All"]
    write_table(version, "table5_repair_by_program_and_order.tex",
                latex_table(header, rows, midrule_before_last=True))


def histograms(df, by, panels, title, version, name, ncols):
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    bins = np.linspace(0, 100, 51)
    for ax, key in zip(axes.flat, panels):
        d = df[df[by] == key]
        before, after = 100 * d["before"], 100 * d["test_accuracy"]
        ax.hist(before, bins=bins, color="tab:red", alpha=0.7, edgecolor="black", label=f"Before {METHOD_LABEL}")
        ax.axvline(before.median(), color="darkred", ls="--", label=f"Median (Before): {before.median():.2f}%")
        ax.hist(after, bins=bins, color="tab:green", alpha=0.7, edgecolor="black", label=f"After {METHOD_LABEL}")
        ax.axvline(after.median(), color="darkgreen", ls="--", label=f"Median (After): {after.median():.2f}%")
        label = key if by == "program" else f"Mutation Order {key}"
        ax.set_title(f"{label}\n(Models: Before={len(d)}, After={len(d)})")
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, version, name)


# --------------------------------------------------------------------------- v2

BASELINE_LABELS = {"gp": "GP", "bfs": "BFS"}
#: (a, b) pairs compared on the same mutants: each start against Xavier and against the
#: compiled weights, and GBPR's starts against the symbolic baselines.
PAIRS = [("compiled", "xavier"), ("rescaled", "xavier"), ("soft:10", "xavier"),
         ("rescaled", "compiled"), ("soft:10", "compiled")]


def method_label(m: str) -> str:
    return m if m in BASELINE_LABELS.values() else f"{METHOD_LABEL} ({m})"


def baseline_runs(version: str, budget: int) -> dict[str, pd.DataFrame]:
    """GP/BFS outcomes per mutant at `budget`, as runs tables (arm = "GP"/"BFS")."""
    out, meta = {}, mutant_table(version)
    for m, label in BASELINE_LABELS.items():
        runs_path, hist_path = records.table_paths(results_dir(version, BASELINES), m, BASELINE_BUDGET)
        if not runs_path.exists():
            continue
        t = analysis.baseline_at_budget(pd.read_csv(runs_path), pd.read_csv(hist_path), budget)
        out[label] = t.merge(meta, on="job_id").assign(arm=label)
    return out


def cell(r) -> str:
    return f"{r.pass1:.1f} [{r.ci_lo:.1f}, {r.ci_hi:.1f}]"


def pass1_table(pm, by, keys, methods, version, name, key_label):
    t = analysis.pass_at_1(pm, by=by, methods=methods)
    rows = []
    for m in methods:
        tm = t[t["method"] == m].set_index(by)
        rows.append([method_label(m), *[cell(tm.loc[k]) if k in tm.index else "--" for k in keys],
                     cell(tm.loc["All"])])
    write_table(version, name, latex_table(["Method", *map(key_label, keys), "All"], rows))
    return t


def v2(df, version):
    df = df.assign(arm=df["init"].map(analysis.init_arm))
    starts = [a for a in V2_INITS if a in set(df["arm"])] + sorted(set(df["arm"]) - set(V2_INITS))
    base = baseline_runs(version, BASELINE_BUDGET)
    cols = ["arm", "job_id", "seed", "program", "mutation_order", "test_accuracy"]
    pm = analysis.per_mutant(pd.concat([df[cols], *[b[cols] for b in base.values()]]))
    methods = starts + list(base)
    print(f"v2: {df['job_id'].nunique()} mutants; starts {starts} "
          f"(seeds {sorted(df['seed'].unique())}); baselines {list(base) or 'none'}")

    pass1_table(pm, "program", list(PROGRAMS), methods, version, "v2_pass1_by_program.tex", tt)
    orders = sorted(pm["mutation_order"].unique())
    by_order = pass1_table(pm, "mutation_order", orders, methods, version, "v2_pass1_by_order.tex",
                           lambda o: f"Order {o}")

    s = analysis.seed_agreement(pm, starts)
    write_table(version, "v2_seed_agreement.tex", latex_table(
        ["Start", "Seeds", "None", "Some", "All", "pass@1", "pass@n"],
        [[r.method, r.seeds, pct(r.none), pct(r.some), pct(r.all), pct(r.pass1), pct(r.pass_n)]
         for r in s.itertuples()]))

    pairs = [(a, b) for a, b in PAIRS if a in methods and b in methods]
    pairs += [(a, b) for a in starts for b in base]
    rows = []
    for a, b in pairs:
        d = analysis.paired_difference(pm, a, b, by="program").set_index("program")
        rows.append([f"{method_label(a)} $-$ {method_label(b)}",
                     *[f"{d.loc[p, 'diff']:+.1f} [{d.loc[p, 'ci_lo']:+.1f}, {d.loc[p, 'ci_hi']:+.1f}]"
                       if p in d.index else "--" for p in [*PROGRAMS, "All"]],
                     f"{d.loc['All', 'a_better']}/{d.loc['All', 'b_better']}",
                     f"{d.loc['All', 'sign_p']:.2g}"])
    write_table(version, "v2_paired.tex", latex_table(
        ["Difference (points)", *map(tt, PROGRAMS), "All", "Better (a/b)", "Sign test $p$"], rows))

    tr = analysis.order_trend(pm, methods)
    write_table(version, "v2_order_trend.tex", latex_table(
        ["Method", "Mutants", "pass@1 change per order (points) [95\\% CI]"],
        [[method_label(r.method), r.mutants, f"{r.slope:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]"]
         for r in tr.itertuples()]))

    if base:
        rows = []
        for b in BASELINE_BUDGETS:
            at = baseline_runs(version, b)
            pmb = analysis.per_mutant(pd.concat([t[cols] for t in at.values()]))
            t = analysis.pass_at_1(pmb).set_index("method")
            rows.append([b, *[cell(t.loc[m]) if m in t.index else "--" for m in base]])
        write_table(version, "v2_baseline_budgets.tex",
                    latex_table(["Budget (programs)", *base], rows))

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(methods):
        t = by_order[(by_order["method"] == m) & (by_order["mutation_order"] != "All")]
        x = t["mutation_order"].astype(int) + (i - (len(methods) - 1) / 2) * 0.06
        ax.errorbar(x, t["pass1"], yerr=[t["pass1"] - t["ci_lo"], t["ci_hi"] - t["pass1"]],
                    marker="o", capsize=3, label=method_label(m))
    ax.set_xticks(orders)
    ax.set_xlabel("Mutation order")
    ax.set_ylabel("pass@1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    save_figure(fig, version, "v2_pass1_by_order.pdf")


@click.command()
@click.option("--version", default="v1_tmlr", show_default=True)
def main(version):
    df = load_runs(version, "main_repair")
    df = df[df["program"].isin(PROGRAMS)].assign(before=lambda d: before_accuracy(d))
    if df["job_id"].duplicated().any():  # v2: several starts and seeds per mutant
        v2(df, version)
        return
    print(f"{len(df)} mutants of {sorted(df['program'].unique())}")
    table1(df, version)
    table5(df, version)
    histograms(df, "program", list(PROGRAMS),
               f"Distribution of Correctness Accuracy by Program (Before and After {METHOD_LABEL})",
               version, "fig3_accuracy_by_program.pdf", ncols=2)
    histograms(df, "mutation_order", sorted(df["mutation_order"].unique()),
               f"Distribution of Correctness Accuracy by Mutation Order (Before and After {METHOD_LABEL})",
               version, "fig7_accuracy_by_order.pdf", ncols=3)


if __name__ == "__main__":
    main()
