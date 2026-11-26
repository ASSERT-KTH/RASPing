"""Submitting jobs to SLURM (via submitit), each running inside the apptainer container.

Every launcher builds a list of command lines (e.g. `python -m rasping.repair ...`)
and calls `submit(commands, name=...)`. Use `--local` (or `executor="local"`) to run
them in-process for a smoke test, and `dry_run=True` to only print them.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Sequence

from rasping.config import REPO_ROOT

CONTAINER = REPO_ROOT / "container.sif"
LOG_DIR = REPO_ROOT / "submitit_logs"

#: Cluster defaults used for the v1 runs: one MIG slice (1/7 A100, 10 GB) per job.
#: Largest SLURM job array a submission uses. Berzelius rejects arrays of more than 1,000
#: tasks (MaxArraySize 1001). Its account also caps queued jobs at 5,000 (GrpSubmitJobs),
#: shared by everyone on the account, so very large launches must be fed in over time.
MAX_ARRAY_SIZE = 1000

DEFAULT_SLURM = {
    "timeout_min": 12 * 60,
    "slurm_additional_parameters": {"reservation": "1g.10gb"},
}


def container_cmd(cmd: Sequence[str], gpu: bool = True) -> list[str]:
    """Wrap a command so it runs inside the project container from the repo root."""
    # The image holds only the locked dependencies; the checkout is put on PYTHONPATH.
    return ["apptainer", "exec", *(["--nv"] if gpu else []), "--pwd", str(REPO_ROOT),
            "--env", f"PYTHONPATH={REPO_ROOT}", str(CONTAINER), *cmd]


def run_command(cmd: Sequence[str]) -> str:
    """Run one command, raising with its output on failure (executed on the worker)."""
    proc = subprocess.run(list(cmd), capture_output=True, text=True, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {shlex.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout[-5000:]}\n--- stderr ---\n{proc.stderr[-5000:]}"
        )
    return proc.stdout[-2000:]


def submit(
    commands: Sequence[Sequence[str]],
    name: str,
    executor: str = "slurm",
    container: bool = True,
    gpu: bool = True,
    dry_run: bool = False,
    array_parallelism: int = 256,
    **slurm_params,
):
    """Submit one job per command. Returns the submitit jobs (or [] for a dry run)."""
    cmds = [container_cmd(c, gpu) if container else list(c) for c in commands]
    if dry_run:
        for c in cmds:
            print(shlex.join(c))
        print(f"[dry run] {len(cmds)} jobs")
        return []
    if not cmds:
        print("Nothing to submit")
        return []

    import submitit

    folder = LOG_DIR / name
    if executor == "local":
        ex = submitit.LocalExecutor(folder=folder)
        ex.update_parameters(timeout_min=slurm_params.get("timeout_min", 60))
    else:
        ex = submitit.AutoExecutor(folder=folder)
        params = {**DEFAULT_SLURM, **slurm_params}
        ex.update_parameters(name=name, nodes=1, slurm_array_parallelism=array_parallelism,
                             **params)
    jobs = []
    for start in range(0, len(cmds), MAX_ARRAY_SIZE):  # one SLURM array per chunk
        with ex.batch():
            jobs += [ex.submit(run_command, c) for c in cmds[start:start + MAX_ARRAY_SIZE]]
    print(f"Submitted {len(jobs)} jobs ({name}) in {-(-len(cmds) // MAX_ARRAY_SIZE)} arrays; "
          f"logs in {folder}")
    return jobs


def add_cli_options(f):
    """Click options shared by all launchers."""
    import click

    for opt in reversed([
        click.option("--dry-run", is_flag=True, help="Print the commands, submit nothing"),
        click.option("--local", is_flag=True, help="Run with submitit's LocalExecutor"),
        click.option("--no-container", is_flag=True, help="Run commands outside apptainer"),
        click.option("--limit", type=int, default=None, help="Submit at most this many jobs"),
    ]):
        f = opt(f)
    return f


def submit_from_cli(commands, name, dry_run, local, no_container, limit, **kw):
    commands = list(commands)[:limit] if limit else list(commands)
    return submit(commands, name, executor="local" if local else "slurm",
                  container=not no_container, dry_run=dry_run, **kw)


def pending(run_dirs: Sequence[Path]) -> list[Path]:
    """Run directories that do not have a finished run.json yet (for resubmission)."""
    return [d for d in run_dirs if not (Path(d) / "run.json").exists()]
