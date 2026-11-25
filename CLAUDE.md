# CLAUDE.md

Research code for gradient-based repair of Tracr-compiled RASP programs. See README.md for the
repository map and how each paper artifact is produced.

## Commands

- `uv sync` installs the environment. Always run Python through `uv run` or `.venv/bin/python`.
- `uv run pytest` runs the fast suite (~2 min). `uv run pytest --slow` adds the tests that compile every mutant.
- `uv run python scripts/reproduce_paper.py --version v1_tmlr` regenerates all tables and figures.
- `python -m rasping.repair --job-id <id> --out runs/...` repairs one mutant.

## Rules

- **Constants live in `rasping/config.py`.** This covers the threshold, program lists, max length, training defaults and paths. Never redefine them in scripts.
- **Program names.** Canonical names are `hist most_freq reverse sort` (`config.PROGRAMS`). Data files use the historical `most-freq`. Map between them with `rasping.programs.canonical()` and `.data_name`; never hand-roll the mapping.
- **Two benchmarks, one version each.**
  - RaspBugs is the 847 mutants in `raspbugs/raspbugs.jsonl.gz`; there are no subsets or versions. The v1 results tables still contain Dyck rows, so filter by `config.PROGRAMS` when analysing them.
  - TracrBench mutants are `tracrbench/tracrbench.jsonl.gz`, over the 77 programs that pass the fixed inclusion rules in `config.TracrBenchRules`; see `tracrbench/README.md`. Don't change the rules after seeing results. TracrBench programs compile with `mlp_exactness=1000` (`Program.mlp_exactness`); never hard-code 120.
- **Mutants** come from the manifests (`rasping.raspbugs`, `rasping.tracrbench`, or `rasping.benchmarks.get_mutant` for either), never from raw cosmic-ray output.
- **Never commit raw run outputs.** This covers `model.npy`, `trajectory.pkl` and anything under `runs/`. Only tidy tables go in `results/<version>/`, and only generated artifacts go in `paper_artifacts/<version>/`.
- **`results/v1_tmlr/` is frozen.** It holds the submitted paper's data. `tests/test_paper_numbers.py` pins it, so never edit it. New results go in a new version directory, e.g. `results/v2/`.
- **Behaviour changes must be visible.** `tests/test_golden.py` and `tests/test_golden_search.py` pin the v1 numerics. Late training epochs are only bit-exact on the machine that recorded them, so by default they are compared loosely; `pytest --golden-exact` compares them fully (F34). If a change is meant to alter behaviour, re-record the golden file in the same commit and add or update an entry in `docs/FINDINGS.md`. That entry gives the evidence, the fix commit, and which results and artifacts it invalidates.
- **Experiment scripts** are `experiments/<name>/{launch,analyze}.py`.
  - Launchers build `python -m ...` command lines and submit them with `rasping.cluster`; they must support `--dry-run`.
  - Analyses take `--version` and write through `rasping.tables`.
  - Put computations worth testing in `rasping/analysis.py`.
- **Log every new bug or paper/code inconsistency** in `docs/FINDINGS.md`, even if you don't fix it.
- **JAX config.** Importing `rasping` enables x64 and float32 matmuls, which compiled models need for exactness (`rasping/_jax_setup.py`).
