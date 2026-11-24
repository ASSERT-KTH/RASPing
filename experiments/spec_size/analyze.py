"""Specification-size ablation: Table 8 and Figure 8 (paper App. D).

    python experiments/spec_size/analyze.py --version v1_tmlr

Restricted to the main-evaluation mutants. If the spec_size runs have no N=40,000
point, the main runs are used for it (as in v1; F8). Baselines: the
`<method>_budget<B>_runs.csv` tables in results/<version>/spec_size/ (v1: budget 500)
or, failing that, in symbolic_baselines/; their per-mutant score is the best test
accuracy over all candidates (F17).
"""

import click
import matplotlib.pyplot as plt
import numpy as np

from rasping import analysis
from rasping.config import MAIN_PROGRAMS, METHOD_LABEL
from rasping.results import load_runs, results_dir
from rasping.tables import latex_table, save_figure, write_table


def load(version, budget):
    main = load_runs(version, "main_repair")
    main = main[main["program"].isin(MAIN_PROGRAMS)]
    ref = set(main["job_id"])
    spec = load_runs(version, "spec_size")
    spec = spec[spec["job_id"].isin(ref)]
    gbpr = {int(n): d for n, d in spec.groupby("n_train")}
    if 40000 not in gbpr:  # v1: the main runs stood in for N=40,000 (F8)
        gbpr[40000] = main
    base = {}
    for m in ("gp", "bfs"):
        for exp in ("spec_size", "symbolic_baselines"):
            if (results_dir(version, exp) / f"{m}_budget{budget}_runs.csv").exists():
                b = load_runs(version, exp, f"{m}_budget{budget}_runs.csv")
                base[m] = b[b["job_id"].isin(ref)]
                break
    return gbpr, base


def table8(t, version):
    best = t.loc[:, t.columns.str.startswith("N=")]
    rows = []
    for o, r in t.iterrows():
        sym = max(r["GP"], r["BFS"])
        cells = []
        for c, v in r.items():
            s = f"{v:.1f}"
            if c.startswith("N=") and v > sym:
                s = rf"\textbf{{{s}}}"
            if v == r.max():
                s = rf"\underline{{{s}}}"
            cells.append(s)
        rows.append([{1: "1 (simplest)", 5: "5 (hardest)"}.get(o, o), *cells])
    header = ["Mutation order", *[c.replace("N=", "N=") for c in best.columns], "GP", "BFS"]
    write_table(version, "table8_spec_size.tex", latex_table(header, rows, midrule_before_last=True))


def figure8(t, version):
    ns = [int(c[2:]) for c in t.columns if c.startswith("N=")]
    orders = [o for o in t.index if o != "All"]
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(orders)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    for o, c in zip(orders, colors):
        y = [t.loc[o, f"N={n}"] for n in ns]
        a.plot(ns, y, "o-", color=c, label=f"Order {o}")
        b.plot(ns, np.array(y) - max(t.loc[o, "GP"], t.loc[o, "BFS"]), "o-", color=c, label=f"Order {o}")
    b.axhline(0, color="black", ls="--", label="Symbolic baseline")
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{n // 1000}k" if n >= 1000 else str(n) for n in ns], rotation=45)
        ax.set_xlabel("Specification size $N$")
        ax.grid(True, alpha=0.3)
        ax.legend()
    a.set_ylabel("Pass rate (%)")
    a.set_ylim(0, 100)
    b.set_ylabel(f"{METHOD_LABEL} $-$ best symbolic (pp)")
    fig.tight_layout()
    save_figure(fig, version, "fig8_spec_size.pdf")


@click.command()
@click.option("--version", default="v1_tmlr", show_default=True)
@click.option("--budget", type=int, default=None,
              help="Baseline budget [default: 500 for v1_tmlr (as submitted), else 1000 (F8)]")
def main(version, budget):
    budget = budget or (500 if version == "v1_tmlr" else 1000)
    gbpr, base = load(version, budget)
    t = analysis.spec_size_table(gbpr, base)
    print(t.round(1).to_string())
    table8(t, version)
    figure8(t, version)


if __name__ == "__main__":
    main()
