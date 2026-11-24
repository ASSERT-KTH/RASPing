"""Imbalance-aware metrics for the Dyck mutants (paper App. C, Tables 6-7).

Needs raw outputs (repaired weights / searched programs), so it runs after the main
repair and symbolic-baseline runs and writes results/<version>/dyck/:

    python experiments/dyck/compute_metrics.py --version v2 --method gbpr
    python experiments/dyck/compute_metrics.py --version v2 --method gp --budget 1000
    python experiments/dyck/compute_metrics.py --version v2 --method bfs --budget 1000

GBPR: the repaired model on the test split. GP/BFS: the best-by-fitness program
(`best_source`) executed symbolically on the test split.
"""

import json
from pathlib import Path

import click
import jax.numpy as jnp
import numpy as np
import pandas as pd

from rasping import data, programs, raspbugs
from rasping.config import DYCK_PROGRAMS, MAX_LEN, RUNS_DIR
from rasping.metrics import dyck_metrics
from rasping.results import results_dir


def gbpr_metrics(job_id: str, run_dir: Path) -> dict:
    m = raspbugs.get(job_id)
    model = m.compile()
    model.load(run_dir / "model.npy")
    X, Y = (np.asarray(a) for a in data.encode(data.load(m.program, "test"), model))
    true, pred = [], []
    for s in range(0, len(X), 500):
        x, y = X[s:s + 500], Y[s:s + 500]
        p = np.asarray(jnp.argmax(model.forward(jnp.array(x)).unembedded_output, axis=-1))
        valid = (x != model.pad_token)
        valid[:, 0] = False
        true += [y[i][valid[i]] for i in range(len(x))]
        pred += [p[i][valid[i]] for i in range(len(x))]
    return {"program_name": m.program, "job_id": job_id, **dyck_metrics(true, pred)}


def baseline_metrics(record: dict) -> dict:
    prog = programs.canonical(record["program_name"])
    test = data.load(prog, "test")
    sop = programs.build_sop(record["best_source"], prog, MAX_LEN)  # v1 used 11 (F11)
    true, pred = [], []
    for x, y in test:
        y = np.array(y[1:])
        try:
            out = list(sop(x[1:]))
        except Exception:
            out = [None] * len(y)
        p = np.array([v if v in (0, 1) else -1 for v in out]) if len(out) == len(y) else np.full_like(y, -1)
        true.append(y)
        pred.append(p)
    return {"program_name": prog, "job_id": record["job_id"], "stored_test_acc": record.get("test_acc"),
            **dyck_metrics(true, pred)}


@click.command()
@click.option("--version", required=True)
@click.option("--method", type=click.Choice(["gbpr", "gp", "bfs"]), required=True)
@click.option("--budget", default=1000, show_default=True)
def main(version, method, budget):
    rows = []
    for m in raspbugs.mutants("dyck"):
        if method == "gbpr":
            d = RUNS_DIR / "main_repair" / m.program / m.job_id
            if (d / "model.npy").exists():
                rows.append(gbpr_metrics(m.job_id, d))
        else:
            f = RUNS_DIR / "symbolic_baselines" / f"{method}_budget{budget}" / f"{m.job_id}.json"
            if f.exists():
                rec = json.loads(f.read_text())
                if rec.get("best_source"):
                    rows.append({"method": method.upper(), **baseline_metrics(rec)})
    out = results_dir(version, "dyck")
    out.mkdir(parents=True, exist_ok=True)
    name = "gbpr_metrics.csv" if method == "gbpr" else f"{method}_metrics.csv"
    pd.DataFrame(rows).to_csv(out / name, index=False)
    print(f"wrote {out / name} ({len(rows)} mutants of {DYCK_PROGRAMS})")


if __name__ == "__main__":
    main()
