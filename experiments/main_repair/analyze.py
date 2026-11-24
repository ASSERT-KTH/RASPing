"""Main repair results: Table 1, Table 5, Figure 3 and Figure 7 (paper §5.2, App. B).

    python experiments/main_repair/analyze.py --version v1_tmlr

Reads results/<version>/main_repair/runs.csv; writes paper_artifacts/<version>/.
"Before" accuracies: v2 runs record the compiled mutant's accuracy on the full test
split; v1 only has the mutation oracle's accuracy on 50 test samples (F7).
"""

import click
import matplotlib.pyplot as plt
import numpy as np

from rasping import analysis, raspbugs
from rasping.config import PROGRAMS, METHOD_LABEL
from rasping.results import load_runs
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


@click.command()
@click.option("--version", default="v1_tmlr", show_default=True)
def main(version):
    df = load_runs(version, "main_repair")
    df = df[df["program"].isin(PROGRAMS)].assign(before=lambda d: before_accuracy(d))
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
