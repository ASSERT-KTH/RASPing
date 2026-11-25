"""Run cosmic-ray on a benchmark's reference programs (step 1 of the generation pipeline).

For every program and mutation order, cosmic-ray generates up to 200 mutants and runs the
benchmark's oracle on each. Results go to `<root>/results/<program>/`:
    tracr_<program>_order=<k>.{toml,sqlite,jsonl}   config, session, dump (read by step 2)
    report_<program>_order=<k>.html                 cosmic-ray's HTML report

    python raspbugs/run_mutations.py                                          # RaspBugs
    python raspbugs/run_mutations.py --root tracrbench --program sorting --orders 1,2

cosmic-ray mutates the program file in place. Each program therefore runs in its own
temporary copy of `source/` and `oracle/`, so the checkout is never left mutated (e.g. if
a cluster job is killed) and programs can run in parallel. Orders of one program run one
after the other. The oracle is `oracle/test_<program>.py` if that file exists (RaspBugs),
else the parametrized `oracle/test_program.py::test_model[<program>]` (TracrBench).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import click
import toml

HERE = Path(__file__).resolve().parent
MUTATION_LIMIT = 200


def test_command(root: Path, program: str) -> str:
    if (root / "oracle" / f"test_{program}.py").exists():
        return f"pytest -s -p no:cacheprovider oracle/test_{program}.py"
    return f"pytest -s -p no:cacheprovider oracle/test_program.py::test_model[{program}]"


def config(root: Path, program: str, order: int, timeout: float) -> dict:
    return {"cosmic-ray": {
        "module-path": f"source/{program}.py",
        "timeout": timeout,
        "excluded-modules": [],
        "test-command": test_command(root, program),
        "distributor": {"name": "local"},
        # higher-order mutations (cosmic-ray fork): exactly `order` mutations per mutant
        "mutation-order": order,
        "specific-order": order,
        "mutation-limit": MUTATION_LIMIT,
        "disable-overlapping-mutations": True,
    }}


def run_program(args: tuple[Path, str, list[int], float]) -> list[str]:
    """All orders of one program, in a temporary working copy; returns error messages."""
    root, program, orders, timeout = args
    out = root / "results" / program
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    with tempfile.TemporaryDirectory(prefix=f"mut_{program}_") as tmp:
        work = Path(tmp)
        (work / "source").mkdir()
        for f in ["__init__.py", f"{program}.py"]:
            if (root / "source" / f).exists():
                shutil.copy(root / "source" / f, work / "source" / f)
        shutil.copytree(root / "oracle", work / "oracle", ignore=shutil.ignore_patterns("__pycache__"))
        for order in orders:
            stem = f"tracr_{program}_order={order}"
            try:
                cfg = work / f"{stem}.toml"
                cfg.write_text(toml.dumps(config(root, program, order, timeout)))
                db = work / f"{stem}.sqlite"
                subprocess.run(["cosmic-ray", "init", cfg.name, db.name], cwd=work, check=True)
                subprocess.run(["cosmic-ray", "exec", cfg.name, db.name], cwd=work, check=True)
                with open(work / f"{stem}.jsonl", "w") as f:
                    subprocess.run(["cosmic-ray", "dump", db.name], cwd=work, stdout=f, check=True)
                with open(work / f"report_{program}_order={order}.html", "w") as f:
                    subprocess.run(["cr-html", db.name], cwd=work, stdout=f, check=True)
                for name in [cfg.name, db.name, f"{stem}.jsonl", f"report_{program}_order={order}.html"]:
                    shutil.copy(work / name, out / name)
                print(f"{program} order {order}: done", flush=True)
            except subprocess.CalledProcessError as e:
                errors.append(f"{program} order {order}: {e}")
                print(errors[-1], flush=True)
    return errors


@click.command()
@click.option("--root", default=str(HERE), show_default=True,
              help="Benchmark directory with source/ and oracle/ (raspbugs or tracrbench)")
@click.option("--program", "only", multiple=True, help="Only these programs (default: all in source/)")
@click.option("--orders", default="1,2,3,4,5", show_default=True, help="Comma-separated mutation orders")
@click.option("--timeout", type=float, default=60.0, show_default=True, help="Oracle timeout per mutant (s)")
@click.option("--workers", type=int, default=None, help="Programs run in parallel (default: one per CPU)")
@click.option("--clean", is_flag=True, help="Delete the programs' previous results first")
def main(root, only, orders, timeout, workers, clean):
    root = Path(root).resolve()
    orders = [int(o) for o in orders.split(",")]
    names = list(only) or sorted(f.stem for f in (root / "source").glob("*.py") if f.stem != "__init__")
    if clean:
        for n in names:
            shutil.rmtree(root / "results" / n, ignore_errors=True)
    print(f"{len(names)} programs x orders {orders} in {root}")
    with Pool(workers) as pool:
        errors = [e for errs in pool.map(run_program, [(root, n, orders, timeout) for n in names])
                  for e in errs]
    if errors:
        raise SystemExit("Errors:\n" + "\n".join(errors))
    print(f"All {len(names) * len(orders)} runs completed")


if __name__ == "__main__":
    main()
