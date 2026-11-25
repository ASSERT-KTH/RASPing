"""Import the TracrBench programs and build their specifications (step 0 of the pipeline).

Reads TracrBench's `eval_data.json` at the pinned commit (`config.TRACRBENCH`), applies the
inclusion rules (`config.TracrBenchRules`) to each of its 101 programs and writes

    programs.json                 every program: included or not (with the reason), entry
                                  function, ground truth, size, attention, trivial baselines
    source/<name>.py              the RASP source of each included program (mutated in step 1)
    ../data/tracrbench/<name>_{train,val,test}.jsonl.gz
                                  50,000 examples from the ground truth (deduplicated, seed 0,
                                  80/10/10 split), as for RaspBugs

    python tracrbench/import_programs.py [--eval-data eval_data.json] [--workers 8]

Without --eval-data the file is downloaded from GitHub at the pinned commit; its SHA-256 is
checked against `EVAL_DATA_SHA256`. Takes about 6 minutes on 8 cores.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import shutil
import urllib.request
from pathlib import Path

import click

from rasping.config import DATA_DIR, MAX_LEN, TRACRBENCH, TRACRBENCH_DIR

EVAL_DATA_URL = (TRACRBENCH.repo.replace("github.com", "raw.githubusercontent.com")
                 + f"/{TRACRBENCH.commit}/eval_data.json")
EVAL_DATA_SHA256 = "0c067477a344bb2282eb75b6e9e50e2ddb9f5d28e3ff014b6b8c7a7843891c33"
N_EXAMPLES = 50_000
SPEC_SEED = 0
TIMEOUT_S = 1800  # per program (compiling the largest takes minutes)


def _uses_attention(sop) -> bool:
    from tracr.rasp import rasp

    seen, stack = set(), [sop]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (rasp.Aggregate, rasp.SelectorWidth)):
            return True
        stack.extend(node.children)
    return False


def analyse(item: tuple[str, dict]) -> dict:
    """Apply the inclusion rules to one program; returns its record and, if included, its spec."""
    from rasping import (
        _jax_setup,  # noqa: F401  (x64 etc., as for RaspBugs)
        data,
        programs,
    )
    from rasping.model import Model

    key, e = item
    name = key.removeprefix("make_")
    rec = {"name": name, "entry": key, "instruction": e["instruction"],
           "ground_truth": e["ground_truth_function"], "included": False, "reason": None}
    # ending with a newline: cosmic-ray's diffs of a file without one do not apply
    source = programs.TRACRBENCH_HEADER + e["rasp_code"].rstrip() + "\n"
    try:
        sop = getattr(programs.module_from_source(source), key)()
    except Exception as ex:  # rule 1
        rec["reason"] = f"does not run: {type(ex).__name__}: {str(ex)[:120]}"
        return rec
    rec["uses_attention"] = _uses_attention(sop)
    prog = programs.Program(name, name, TRACRBENCH.vocab, lambda m, n, v: getattr(m, key)(),
                            benchmark="tracrbench", mlp_exactness=TRACRBENCH.mlp_exactness,
                            ground_truth=programs._ground_truth(e["ground_truth_function"]))
    try:
        spec = data.generate(prog, MAX_LEN, N_EXAMPLES, remove_duplicates=True, seed=SPEC_SEED,
                             timeout=600)
    except Exception as ex:
        rec["reason"] = f"ground truth fails: {type(ex).__name__}: {str(ex)[:120]}"
        return rec
    train, val, test = data.split_train_val_test(spec)
    rec["n_examples"] = len(spec)
    try:
        model = Model(sop, prog.vocab, MAX_LEN, name, mlp_exactness=prog.mlp_exactness)
    except Exception as ex:  # rule 2
        rec["reason"] = f"does not compile: {type(ex).__name__}: {str(ex)[:120]}"
        return rec
    rec["n_params"] = model.n_params
    rec["n_layers"] = len(model.params)
    rec["test_accuracy"] = float(model.exact_match(test).mean())
    outs = [tuple(y[1:]) for _, y in test]
    values = {v for y in outs for v in y}
    rec["const_baseline"] = max(sum(all(v == c for v in y) for y in outs) for c in values) / len(test)
    rec["identity_baseline"] = sum(tuple(x[1:]) == y for (x, _), y in zip(test, outs)) / len(test)
    if rec["test_accuracy"] < 1.0:  # rule 2
        rec["reason"] = f"compiled program is wrong on {1 - rec['test_accuracy']:.1%} of the test split"
    elif max(rec["const_baseline"], rec["identity_baseline"]) >= TRACRBENCH.max_trivial_accuracy:  # rule 3
        rec["reason"] = (f"trivial: constant {rec['const_baseline']:.0%}, "
                         f"identity {rec['identity_baseline']:.0%} exact match")
    elif rec["n_params"] > TRACRBENCH.max_params:  # rule 4
        rec["reason"] = f"too large: {rec['n_params']:,} parameters"
    else:
        rec["included"] = True
        return {**rec, "_source": source, "_splits": (train, val, test)}
    return rec


def load_eval_data(path: str | None) -> dict:
    if path:
        raw = Path(path).read_bytes()
    else:
        with urllib.request.urlopen(EVAL_DATA_URL, timeout=60) as r:
            raw = r.read()
    digest = hashlib.sha256(raw).hexdigest()
    if EVAL_DATA_SHA256 and digest != EVAL_DATA_SHA256:
        raise ValueError(f"eval_data.json has SHA-256 {digest}, expected {EVAL_DATA_SHA256}")
    print(f"eval_data.json: sha256 {digest}")
    return json.loads(raw)


@click.command()
@click.option("--eval-data", default=None, help="Local copy of eval_data.json (default: download)")
@click.option("--workers", default=8, show_default=True)
@click.option("--only", multiple=True, help="Only these programs (no files written; for checks)")
def main(eval_data, workers, only):
    from rasping import data, programs

    items = sorted(load_eval_data(eval_data).items())
    if only:
        items = [(k, e) for k, e in items if k.removeprefix("make_") in only]
    records = []
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        futures = [(k, pool.apply_async(analyse, ((k, e),))) for k, e in items]
        for k, f in futures:
            try:
                rec = f.get(timeout=TIMEOUT_S)
            except mp.TimeoutError:
                rec = {"name": k.removeprefix("make_"), "entry": k, "included": False,
                       "reason": f"timeout (> {TIMEOUT_S} s)"}
            print(f"{rec['name']:32s} {'IN ' if rec['included'] else 'out'} {rec['reason'] or ''}",
                  flush=True)
            records.append(rec)
    if only:
        return

    src_dir, data_dir = TRACRBENCH_DIR / "source", DATA_DIR / "tracrbench"
    for d in (src_dir, data_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    keys = ["name", "entry", "included", "reason", "uses_attention", "n_params", "n_layers",
            "n_examples", "test_accuracy", "const_baseline", "identity_baseline", "instruction",
            "ground_truth"]
    for rec in records:
        if rec["included"]:
            (src_dir / f"{rec['name']}.py").write_text(rec["_source"])
    programs.TRACRBENCH_PROGRAMS.write_text(json.dumps(
        [{k: r.get(k) for k in keys} for r in records], indent=1) + "\n")
    programs.tracrbench_records.cache_clear()
    programs._tracrbench.cache_clear()
    for rec in records:
        if rec["included"]:
            for split, part in zip(("train", "val", "test"), rec["_splits"]):
                data.save(rec["name"], split, part)
    inc = [r for r in records if r["included"]]
    print(f"\n{len(inc)}/{len(records)} programs included "
          f"({sum(r['uses_attention'] for r in inc)} with attention)")


if __name__ == "__main__":
    main()
