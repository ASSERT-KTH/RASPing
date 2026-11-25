"""Experiment entry points: launchers build valid commands, analyses run end to end."""

import importlib.util
import json
import subprocess
import sys

import pytest

from rasping.config import REPO_ROOT, RESULTS_DIR

LAUNCHERS = [
    "experiments/main_repair/launch.py",
    "experiments/spec_size/launch.py",
    "experiments/init_control/launch.py",
    "experiments/init_pilot/launch.py",
    "experiments/trajectories/launch.py",
    "experiments/symbolic_baselines/launch.py --method gp",
    "experiments/symbolic_baselines/launch.py --method bfs",
]


def load_script(path):
    spec = importlib.util.spec_from_file_location("script", REPO_ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script", LAUNCHERS)
def test_launcher_dry_run(script, tmp_path):
    path, *args = script.split()
    out = subprocess.run([sys.executable, str(REPO_ROOT / path), *args, "--dry-run", "--limit", "2"],
                         capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "[dry run]" in out.stdout


def test_init_control_selection_reproduces_v1():
    sel = load_script("experiments/init_control/launch.py").select_mutants
    v1 = json.loads((RESULTS_DIR / "v1_tmlr/init_control/selected_jobs.json").read_text())["selection"]
    assert {p: sel(p) for p in v1} == v1


def test_reproduce_paper_v1(tmp_path):
    out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/reproduce_paper.py"), "--version", "v1_tmlr"],
                         capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    # The committed artifacts must be exactly what the scripts produce (PDFs carry no date).
    diff = subprocess.run(["git", "status", "--porcelain", "paper_artifacts/v1_tmlr"],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    assert diff.returncode == 0 and diff.stdout == "", "regenerated artifacts differ:\n" + diff.stdout


def test_submitit_executor_loads(tmp_path):
    """Dry runs never import submitit; real submissions do (setuptools 82 broke it)."""
    import submitit

    assert submitit.AutoExecutor(folder=tmp_path) is not None
