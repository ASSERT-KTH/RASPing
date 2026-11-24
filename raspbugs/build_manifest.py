"""Aggregate cosmic-ray outputs into the RaspBugs manifest (step 2 of the pipeline).

Step 1, `run_mutations.py`, writes `results/<program>/tracr_<program>_order=<k>.jsonl`:
one line per mutant with the mutation and the oracle's pytest output. This script
re-applies each diff to the reference source, labels the mutant from the oracle output
and writes

    raspbugs.jsonl.gz   the benchmark: buggy mutants whose compiled output vocabulary can
                        represent the specification's outputs (F6), with source, diff, ...
    mutants.csv         every mutant with its status and `representable` (not tracked)

    python build_manifest.py [--results-dir results]
"""

from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
from pathlib import Path

import click
import pandas as pd

HERE = Path(__file__).resolve().parent

#: Oracle outputs that mean the mutation itself is broken (v1 rules; the line numbers
#: refer to assertions in the v1 sources).
FAILED_MUTATION_MARKERS = [
    "SyntaxError", "TypeError", "AttributeError", "IndexError",
    "rasp.py:388: AssertionError",
]


def apply_diff(source: str, diff: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        src, patch = Path(tmp) / "src.py", Path(tmp) / "diff.patch"
        src.write_text(source)
        patch.write_text(diff)
        subprocess.run(["patch", "-F", "3", str(src), str(patch)], capture_output=True, check=True)
        return src.read_text()


def label(result: dict) -> dict:
    """Status of a mutant from the oracle's pytest output (v1 rules)."""
    if result.get("output") is None:
        return {"status": "UNKNOWN_FAILURE"}
    out = result["output"].split("=" * 35 + " FAILURES " + "=" * 35)[0]
    if any(m in out for m in FAILED_MUTATION_MARKERS):
        return {"status": "FAILED_MUTATION"}
    if "timeout" in out:
        return {"status": "TIMEOUT"}
    if "UNCOMPILABLE MODEL" in out:
        return {"status": "UNCOMPILABLE_MODEL"}
    if "MODEL IS NOT BUGGY" in out:
        return {"status": "CORRECT_MODEL"}
    if "Accuracy: " in out:
        return {"status": "BUGGY_MODEL", "accuracy": float(out.split("Accuracy: ")[1].split("\n")[0])}
    raise ValueError(f"Unknown oracle output: {out[:500]}")


def is_representable(program: str, source: str) -> bool:
    from rasping import data, programs
    from rasping.config import MAX_LEN
    from rasping.model import Model

    p = programs.get(program)
    model = Model(programs.build_sop(source, p.name, MAX_LEN), p.vocab, MAX_LEN, p.name)
    return not data.unrepresentable_outputs(p.name, model)


def iter_mutants(results_dir: Path):
    for f in sorted(results_dir.glob("**/tracr_*.jsonl")):
        program = f.parent.name
        reference = (HERE / "source" / f"{program}.py").read_text()
        for line in f.read_text().splitlines():
            mutation, result = json.loads(line)
            diff = result.get("diff")
            yield {
                "job_id": mutation["job_id"],
                "program": program,
                "mutation_order": len(mutation["mutations"]),
                "operators": [m["operator_name"].replace("operators/", "") for m in mutation["mutations"]],
                "diff": diff,
                "source": apply_diff(reference, diff) if diff is not None else None,
                "mutations": mutation["mutations"],
                **{k if k == "status" else "oracle_accuracy_50": v for k, v in label(result).items()},
            }


@click.command()
@click.option("--results-dir", default=str(HERE / "results"), show_default=True)
def main(results_dir):
    rows = list(iter_mutants(Path(results_dir)))
    for r in rows:
        r["representable"] = (r["status"] == "BUGGY_MODEL"
                              and is_representable(r["program"], r["source"]))
    benchmark = [r for r in rows if r["representable"]]
    with gzip.GzipFile(HERE / "raspbugs.jsonl.gz", "wb", mtime=0) as f:
        for r in benchmark:
            f.write((json.dumps({k: r.get(k) for k in (
                "job_id", "program", "mutation_order", "operators", "oracle_accuracy_50",
                "diff", "source", "mutations")}) + "\n").encode())
    table = pd.DataFrame([{k: r.get(k) for k in ("job_id", "program", "mutation_order", "status",
                                                  "oracle_accuracy_50", "representable")}
                          for r in rows])
    table["operators"] = [";".join(r["operators"]) for r in rows]
    table.to_csv(HERE / "mutants.csv", index=False)
    print(table["status"].value_counts().to_string())
    print(f"benchmark (buggy and representable): {len(benchmark)} mutants")


if __name__ == "__main__":
    main()
