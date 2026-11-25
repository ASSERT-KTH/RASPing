"""Aggregate cosmic-ray outputs into a benchmark manifest (step 2 of the pipeline).

Step 1, `run_mutations.py`, writes `<root>/results/<program>/tracr_<program>_order=<k>.jsonl`:
one line per mutant with the mutation and the oracle's pytest output. This script
re-applies each diff to the reference source, labels the mutant from the oracle output
and writes, for the benchmark in `<root>` (raspbugs or tracrbench):

    <root>/<root>.jsonl.gz  the benchmark: buggy mutants whose compiled output vocabulary can
                            represent the specification's outputs (F6), with source, diff, ...
                            TracrBench: at most `config.TRACRBENCH.mutants_per_program` per
                            program, stratified by mutation order (`selected`)
    <root>/mutants.csv      every mutant with its status, `representable` and `selected`

    python raspbugs/build_manifest.py [--root raspbugs] [--results-dir <root>/results]
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
        rec = {"status": "BUGGY_MODEL", "accuracy": float(out.split("Accuracy: ")[1].split("\n")[0])}
        if "REPRESENTABLE: " in out:  # TracrBench's oracle checks it while the model is compiled
            rec["oracle_representable"] = out.split("REPRESENTABLE: ")[1].split("\n")[0].strip() == "True"
        return rec
    raise ValueError(f"Unknown oracle output: {out[:500]}")


def is_representable(program: str, source: str) -> bool:
    from rasping import data, programs
    from rasping.config import MAX_LEN
    from rasping.model import Model

    p = programs.get(program)
    model = Model(programs.build_sop(source, p.name, MAX_LEN), p.vocab, MAX_LEN, p.name,
                  mlp_exactness=p.mlp_exactness)
    return not data.unrepresentable_outputs(p.name, model)


def iter_mutants(root: Path, results_dir: Path):
    for f in sorted(results_dir.glob("**/tracr_*.jsonl")):
        program = f.parent.name
        reference = (root / "source" / f"{program}.py").read_text()
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
                **{{"accuracy": "oracle_accuracy_50"}.get(k, k): v for k, v in label(result).items()},
            }


def select(rows: list[dict], name: str) -> None:
    """Mark the benchmark's mutants (`selected`): all representable ones for RaspBugs; for
    TracrBench at most `mutants_per_program` per program, stratified by mutation order."""
    from rasping.benchmarks import stratified_ids
    from rasping.config import TRACRBENCH

    for r in rows:
        r["selected"] = r["representable"]
    if name != "tracrbench":
        return
    for program in sorted({r["program"] for r in rows}):
        cands = [(r["job_id"], r["mutation_order"]) for r in rows
                 if r["program"] == program and r["representable"]]
        keep = set(stratified_ids(cands, TRACRBENCH.mutants_per_program, TRACRBENCH.sample_seed))
        for r in rows:
            if r["program"] == program:
                r["selected"] = r["job_id"] in keep


@click.command()
@click.option("--root", default=str(HERE), show_default=True, help="Benchmark directory")
@click.option("--results-dir", default=None, help="cosmic-ray outputs (default: <root>/results)")
def main(root, results_dir):
    root = Path(root).resolve()
    rows = list(iter_mutants(root, Path(results_dir) if results_dir else root / "results"))
    for r in rows:
        from_oracle = r.pop("oracle_representable", None)  # TracrBench; RaspBugs: compile here
        r["representable"] = r["status"] == "BUGGY_MODEL" and (
            from_oracle if from_oracle is not None else is_representable(r["program"], r["source"]))
    select(rows, root.name)
    benchmark = [r for r in rows if r["selected"]]
    with gzip.GzipFile(root / f"{root.name}.jsonl.gz", "wb", mtime=0) as f:
        for r in benchmark:
            f.write((json.dumps({k: r.get(k) for k in (
                "job_id", "program", "mutation_order", "operators", "oracle_accuracy_50",
                "diff", "source", "mutations")}) + "\n").encode())
    table = pd.DataFrame([{k: r.get(k) for k in ("job_id", "program", "mutation_order", "status",
                                                  "oracle_accuracy_50", "representable", "selected")}
                          for r in rows])
    table["operators"] = [";".join(r["operators"]) for r in rows]
    table.to_csv(root / "mutants.csv", index=False)
    print(table["status"].value_counts().to_string())
    print(f"buggy and representable: {int(table['representable'].sum())}; "
          f"benchmark: {len(benchmark)} mutants")


if __name__ == "__main__":
    main()
