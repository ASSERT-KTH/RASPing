"""Weight statistics for Table 9: compiled vs repaired vs trained-from-random weights.

Needs the raw run directories (model.npy), so it runs after the init-control and main
repair runs and writes results/<version>/init_control/weight_stats.csv:

    python experiments/init_control/weight_stats.py --version v2

For each selected mutant: the compiled (buggy) weights, the weights after repair from
them (main run) and after training from a random init (each seed).
"""

import json

import click
import numpy as np
import pandas as pd

from rasping import raspbugs
from rasping.config import RUNS_DIR
from rasping.results import results_dir


def vector_stats(params: dict) -> dict:
    flat = np.concatenate([np.asarray(t).ravel() for layer in params.values() for t in layer.values()])
    return {"zero_frac": float((flat == 0).mean()), "std": float(flat.std()),
            "max_abs": float(np.abs(flat).max()), "n_params": int(flat.size)}


def load_params(path):
    return np.load(path, allow_pickle=True).item()


@click.command()
@click.option("--version", required=True)
@click.option("--runs", "runs_root", default=str(RUNS_DIR), show_default=True)
def main(version, runs_root):
    from pathlib import Path

    root = Path(runs_root)
    sel = json.loads((root / "init_control" / "selected_jobs.json").read_text())
    rows = []
    for program, job_ids in sel["selection"].items():
        for job_id in job_ids:
            base = {"job_id": job_id, "program": program}
            rows.append({**base, "state": "compiled", "seed": None,
                         **vector_stats(raspbugs.compile_mutant(job_id).params)})
            p = root / "main_repair" / program / job_id / "model.npy"
            if p.exists():
                rows.append({**base, "state": "repaired_from_compiled", "seed": None,
                             **vector_stats(load_params(p))})
            for d in sorted((root / "init_control" / program / job_id).glob("*_seed_*")):
                if (d / "model.npy").exists():
                    rows.append({**base, "state": f"trained_from_{d.name.split('_seed_')[0]}",
                                 "seed": int(d.name.split("_seed_")[1]),
                                 **vector_stats(load_params(d / "model.npy"))})
    out = results_dir(version, "init_control") / "weight_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.10g")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
