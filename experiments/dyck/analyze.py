"""Dyck programs under imbalance-aware metrics: Table 6 and Table 7 (paper App. C).

    python experiments/dyck/analyze.py --version v1_tmlr

Reads results/<version>/dyck/{gbpr_metrics.csv, gp_metrics.csv, bfs_metrics.csv}
(v1: GP and BFS share baseline_metrics.csv). The "Majority" column is the accuracy of
always answering "unbalanced", computed from the committed test splits.
"""

import click
import numpy as np
import pandas as pd

from rasping import analysis, data
from rasping.config import DYCK_PROGRAMS, METHOD_LABEL
from rasping.results import results_dir
from rasping.tables import latex_table, tt, write_table


def majority_baseline(program: str) -> float:
    """% of test sequences that are unbalanced (all-zero outputs)."""
    test = data.load(program, "test")
    return 100.0 * float(np.mean([not any(v == 1 for v in y[1:]) for _, y in test]))


def load(version):
    d = results_dir(version, "dyck")
    frames = [pd.read_csv(d / "gbpr_metrics.csv").assign(method="gbpr")]
    for f in ("baseline_metrics.csv", "gp_metrics.csv", "bfs_metrics.csv"):
        if (d / f).exists():
            frames.append(pd.read_csv(d / f))
    return analysis.dyck_summary(pd.concat(frames)).set_index(["method", "program_name"])


@click.command()
@click.option("--version", default="v1_tmlr", show_default=True)
def main(version):
    s = load(version)
    rows = []
    for p in DYCK_PROGRAMS:
        r = s.loc[("gbpr", p)]
        rows.append([tt(p), int(r.n), f"{majority_baseline(p):.2f}\\%",
                     f"{r.acc:.2f}\\% ({r.acc_fixed:.1f}\\%)", f"{r.recall:.2f}\\% ({r.recall_fixed:.1f}\\%)",
                     f"{r.bal_acc:.2f}\\% ({r.bal_acc_fixed:.1f}\\%)", f"{r.macro_f1:.2f}\\%"])
    write_table(version, "table6_dyck_gbpr.tex", latex_table(
        ["Program", "$n$", "Majority", "Accuracy", "Recall", "Bal. Acc.", "Macro-F1"], rows))

    rows = []
    for p, label in zip(DYCK_PROGRAMS, ("dyck-1", "dyck-2")):
        cells = []
        for m in ("gbpr", "gp", "bfs"):
            r = s.loc[(m, p)]
            cells += [f"{r.acc_fixed:.1f}\\%", f"{r.recall_fixed:.1f}\\%"]
        rows.append([f"{tt(p)} ({label})", int(s.loc[("gbpr", p)].n), *cells])
    tex = latex_table(["Program", "$n$", *["Accuracy", "Recall"] * 3], rows)
    group = (rf" & & \multicolumn{{2}}{{c}}{{{METHOD_LABEL}}} & \multicolumn{{2}}{{c}}{{GP}}"
             r" & \multicolumn{2}{c}{BFS} \\" "\n"
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n")
    write_table(version, "table7_dyck_all_methods.tex",
                tex.replace(r"\toprule" + "\n", r"\toprule" + "\n" + group, 1))
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
