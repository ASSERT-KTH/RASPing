# RASPing: Gradient-Based Program Repair of compiled RASP programs

Code, benchmark and results for *Gradient-Based Program Repair: Fixing Bugs in Continuous
Program Spaces* ([arXiv:2505.17703](https://arxiv.org/abs/2505.17703)). The method works in
four steps:

1. Take a buggy RASP program from the RaspBugs benchmark (847 mutants of four programs).
2. Compile it exactly into Transformer weights with [Tracr](https://github.com/google-deepmind/tracr).
3. Run gradient descent on those weights against an input-output specification.
4. Count the program as repaired if its held-out exact-match accuracy is at least 99%.

The code is compared against symbolic search baselines (genetic programming and BFS).

> **Status (Nov 2026).** The TMLR submission was rejected with an invitation to resubmit. The
> code was cleaned up afterwards. `results/v1_tmlr/` and `paper_artifacts/v1_tmlr/` reproduce
> the submitted paper exactly, except Appendix C: the two Dyck programs were removed from
> RaspBugs, which now has 847 mutants instead of 1,466 (FINDINGS F35). Bugs and
> inconsistencies found during the cleanup are tracked in
> [`docs/FINDINGS.md`](docs/FINDINGS.md); some of them change results once rerun.
> The submitted code is tagged `tmlr-submission-1`.

## Setup

```bash
uv sync                      # Python 3.13, CPU JAX, exact locked versions
uv run pytest                # ~2 min; includes the paper-number and golden-equivalence tests
```

On the cluster, build the container once with `apptainer build container.sif container.def`.
The launchers run every job inside it, on a GPU slice, via SLURM/submitit.

## Reproduce the paper's tables and figures

All tables and figures are computed from the small tidy tables in `results/` (no GPU, no raw runs):

```bash
uv run python scripts/reproduce_paper.py --version v1_tmlr   # -> paper_artifacts/v1_tmlr/
```

| Paper artifact | Script | Inputs (`results/<version>/...`) |
|---|---|---|
| Table 1, Table 5, Fig 3, Fig 7 | `experiments/main_repair/analyze.py` | `main_repair/runs.csv` |
| Fig 4, Fig 5 | `experiments/symbolic_baselines/analyze.py` | `main_repair/val_curves.csv.gz`, `symbolic_baselines/*` |
| Table 8, Fig 8 | `experiments/spec_size/analyze.py` | `spec_size/*`, `main_repair/runs.csv` |
| Table 2, Table 9, Fig 9, Fig 10 | `experiments/init_control/analyze.py` | `init_control/*`, `main_repair/*` |
| Fig 1, Fig 6 | `experiments/trajectories/plot.py` | raw `trajectory.pkl` (archive, or `experiments/trajectories/launch.py`) |
| Table 4, §4 statistics | `tests/test_paper_numbers.py` | `mutants.csv` |

`tests/test_paper_numbers.py` recomputes every number above with independent pandas code and
checks it against the submitted paper.

## Rerun the experiments

Each experiment has a `launch.py`, which submits one SLURM job per mutant and skips finished ones.
Raw runs are written to `runs/` (gitignored). Collect them into tidy tables, then analyze:

```bash
python experiments/main_repair/launch.py --dry-run         # inspect; drop --dry-run to submit
python -m rasping.results collect main_repair --version v2  # runs/ -> results/v2/main_repair/
python experiments/main_repair/analyze.py --version v2
```

| Experiment | Launch | Collect |
|---|---|---|
| main repair (§5.2) | `experiments/main_repair/launch.py` | `python -m rasping.results collect main_repair` |
| GP / BFS (§5.3) | `experiments/symbolic_baselines/launch.py --method gp\|bfs --budget 1000` | `experiments/symbolic_baselines/collect.py` |
| spec size (§5.4) | `experiments/spec_size/launch.py` | `python -m rasping.results collect spec_size` |
| init control (§5.5) | `experiments/init_control/launch.py` | `python -m rasping.results collect init_control` + `experiments/init_control/weight_stats.py` |
| init pilot (v2) | `experiments/init_pilot/launch.py` | `python -m rasping.results collect init_pilot`, then `experiments/init_pilot/analyze.py` |
| trajectories (§5.6) | `experiments/trajectories/launch.py` | — |

To repair a single mutant locally:

```bash
python -m rasping.repair --job-id 2a722d5d5449441fad93f98c73c633a2 --out runs/test/hist
```

Useful options are `--init` (`compiled`, `xavier`, `shuffled`, `rescaled`, `soft:<c>`,
`from:<job_id>`; see `rasping.repair.INITS`), `--seed`, `--n-train`, `--n-steps` and
`--store-trajectory`.

## Repository map

```
rasping/            the package: config, programs, data, model, train, repair, benchmarks (raspbugs, tracrbench),
                    baselines/ (GP, BFS), analysis, results, cluster, tables
raspbugs/           benchmark generation pipeline + manifest raspbugs.jsonl.gz (see its README)
tracrbench/         second benchmark: TracrBench programs, their import and mutants (see its README)
experiments/        one directory per paper experiment: launch.py / analyze.py
data/               input-output specifications (train/val/test JSONL per program)
results/<version>/  tidy result tables (tracked); v1_tmlr = the submitted paper
paper_artifacts/    generated LaTeX tables and PDF figures, per version
scripts/            reproduce_paper.py, generate_data.py
tests/              unit tests, paper-number oracle, golden equivalence with the v1 code
docs/FINDINGS.md    log of bugs / inconsistencies and their fixes
```

## Raw data and old history

Raw run directories (weights, per-run curves, trajectories, raw cosmic-ray outputs) are not
in git. A one-off script derived `results/v1_tmlr/` from them (`scripts/freeze_v1_results.py`,
in git history up to commit 992e2c0).

The repository history was rewritten in Nov 2026 to drop those raw outputs (9 GB → 52 MB).
The original, unrewritten history is still on GitHub under `refs/archive/*`. It includes
every original branch and the original `tmlr-submission-1` tag, with all weights and run
directories. A normal clone does not download it; fetch it explicitly (about 9 GB):

```bash
git fetch origin 'refs/archive/*:refs/archive/*'
git checkout refs/archive/tags/tmlr-submission-1     # the submitted state, with raw runs
```

Contact: André Silva.
