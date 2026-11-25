"""Index the compiled architecture of every mutant -> <benchmark>/architectures.csv.

One row per mutant: job_id, program, mutation_order, n_params, n_layers and `signature`,
a hash of the sorted parameter names and shapes. Mutants with the same signature have
interchangeable weights (almost always mutants of the same program; a few most_freq and
reverse mutants share shapes). That is what the "from:<job_id>" init needs (rasping.repair;
look-ups via `rasping.raspbugs.shape_twins`). The table also answers how architectures vary
within and across programs.

    python raspbugs/architectures.py [--program sort]   (compiles all 847 mutants, ~7 min)
    python raspbugs/architectures.py --benchmark tracrbench
"""

from __future__ import annotations

import hashlib

import click
import pandas as pd

from rasping.benchmarks import BENCHMARKS


def signature(params) -> tuple[str, int, int]:
    """(hash of names and shapes, number of parameters, number of layers)."""
    items = sorted((f"{k}/{s}", tuple(int(d) for d in params[k][s].shape)) for k in params for s in params[k])
    n = sum(int(pd.Series(shape).prod()) if shape else 1 for _, shape in items)
    h = hashlib.sha1(repr(items).encode()).hexdigest()[:16]
    return h, n, len(params)


@click.command()
@click.option("--benchmark", "name", type=click.Choice(sorted(BENCHMARKS)), default="raspbugs",
              show_default=True)
@click.option("--program", "only", multiple=True, help="Only these programs (merged into the CSV)")
def main(name, only):
    bench = BENCHMARKS[name]
    out = bench.architectures_path
    rows = []
    for p in only or bench.programs():
        for m in bench.mutants(program=p):
            h, n, layers = signature(m.compile().params)
            rows.append({"job_id": m.job_id, "program": m.program, "mutation_order": m.mutation_order,
                         "n_params": n, "n_layers": layers, "signature": h})
        print(p, "done", flush=True)
    new = pd.DataFrame(rows)
    if only and out.exists():
        new = pd.concat([bench.architectures()[lambda d: ~d["program"].isin(only)], new])
    new = new.sort_values(["program", "job_id"])
    new.to_csv(out, index=False)
    g = new.groupby("program")
    print(pd.DataFrame({"mutants": g.size(), "architectures": g["signature"].nunique(),
                        "with_twin": g["signature"].apply(lambda s: s.duplicated(keep=False).sum()),
                        "min_params": g["n_params"].min(), "max_params": g["n_params"].max()}).to_string())


if __name__ == "__main__":
    main()
