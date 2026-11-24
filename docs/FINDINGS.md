# Findings log

Bugs, paper↔code inconsistencies and methodological issues found while cleaning up the
repository after the TMLR decision (Nov 2026). Every entry that changes code links the
commit that fixed it and lists which results/paper artifacts it invalidates.

The submitted state of the code is tagged `tmlr-submission-1`; the numbers in the submitted
paper are frozen in `results/v1_tmlr/` and checked by `tests/test_paper_numbers.py`.

**Status legend**
- `open` — found, not yet fixed
- `fixed` — code fixed; affected results need a rerun to change (see *Invalidates*)
- `fixed+rerun` — code fixed and affected results regenerated
- `paper` — no code change needed, but the paper text must be updated
- `wontfix` — documented, deliberately kept (with reason)

Paper references are to the TMLR submission (`paper.pdf`, rejected Nov 2026).

---

## Data and benchmark

### F1 — Dyck-2 specification never contains inputs longer than 8 — `open`
- **Evidence:** `src/functions.py:71` — `generateData(exhaustive=True)` enumerates inputs in
  order of length and returns `data[:size]` *before* shuffling. For `shuffle_dyck2` with
  n=50,000 and max_len=10, every input in `data/shuffle_dyck2_*.jsonl` has length 2–8.
- **Paper:** §4 says input lengths are sampled between 2 and 10.
- **Invalidates:** all `shuffle_dyck2` results (App. C Tables 6–7; spec-size Dyck runs).

### F6 — The 1,466-mutant benchmark is defined by a join with GBPR results — `open`
- **Evidence:** `experiments/mutation/raspbugs.ipynb` inner-joins
  `aggregated_mutations.json` (1,773 `BUGGY_MODEL` rows) with
  `train_mutations/analysis_outputs/all_test_accuracies.csv` (GBPR results) to produce
  `raspbugs.json`. 307 buggy mutants are dropped; ~100 are explained by duplicate sources,
  the rest are undocumented and skew toward reverse/sort/hist at orders 3–5. Every
  experiment script loads `aggregated_mutations.json` instead and re-filters (or doesn't).
- **Invalidates:** benchmark size and composition (§4, every table).

### F7 — "Buggy" label decided on only 50 test samples — `open`
- **Evidence:** `experiments/mutation/tests/test.py:29` loads `data/<prog>_test.jsonl[:50]`.
  The pre-repair accuracy stored in `execution_result.accuracy` (and plotted as "Before GBPR"
  in Figs 3 and 7) is also computed on these 50 samples, whereas post-repair accuracy is on
  the full 5,000-sample test split.
- **Paper:** §4 says a buggy program must be incorrect on the input-output specification.
- **Invalidates:** benchmark membership (mutants correct on the 50 but wrong elsewhere are
  mislabelled `CORRECT_MODEL`); the "before" histograms and the "min/median/mean" statistics
  in §4.

### F19 — Undocumented mutant statuses — `open`
- **Evidence:** `aggregated_mutations.json` has, besides the four statuses described in §4,
  `UNKNOWN_FAILURE` (362) and `TIMEOUT` (78).
- **Fix:** document in the benchmark README / paper, or investigate and reclassify.

## Training (GBPR)

### F2 — Early-stopping patience: paper says 10, code used 50 — `open`
- **Evidence:** `experiments/train_mutations/train_mutations.py:170` CLI default
  `early_stopping_patience=50`; no launcher overrides it (the function default of 10 at :37
  is never reached). `launch_random_init_baseline.py:11-12` confirms 50.
- **Paper:** §5.1 "validation loss improvement < 1e-4 for 10 epochs".

### F8 — Spec-size ablation: N=40,000 point and baselines don't match the stated setup — `open`
- **Evidence:** `experiments/spec_size/launch_spec_size.py:8` runs N ∈ {100,…,25,000} with
  `N_STEPS=500k`; the N=40,000 column reuses the main runs, which have `n_epochs=10000`
  (≈1.56M step cap). The GP/BFS columns use the 500-program budget runs while the main
  experiments use 1,000.
- **Paper:** App. D says the gradient-step budget is fixed at 500k for all N.

## Symbolic baselines (GP / BFS)

### F3 — BFS is told the true mutation order — `open`
- **Evidence:** `experiments/gp_baseline/run_gp_exhaustive.py:330` sets
  `max_depth = mutation_order` of the mutant being repaired.
- **Paper:** not disclosed; this is oracle information on top of the operator oracle §5.3
  already acknowledges.

### F5 — GP fitness uses train+val; paper says train — `open`
- **Evidence:** `experiments/gp_baseline/gp_core.py:375`.

### F9 — Conflicting GP defaults; GP and BFS share an output dir — `open`
- **Evidence:** CLI defaults pop=16, k=3, budget=500 (`run_gp_baseline.py`) vs
  `run_gp_for_bug` defaults pop=60, k=4 (`gp_core.py:675-677`). Both runners default
  `--output-dir` to `experiments/gp_baseline/results` (`run_gp_baseline.py:120`,
  `run_gp_exhaustive.py:392`).

### F17 — Baseline progress curves select on the test set — `open`
- **Evidence:** `experiments/gp_baseline/plot_progress.py:11-78` builds GP/BFS curves from
  `best_test_acc`, the maximum *test* accuracy over all candidates evaluated so far, rather
  than the test accuracy of the candidate with the best training fitness (`test_acc` in the
  result record).
- **Invalidates:** Fig 4, Fig 5, the §5.3 convergence numbers, and baseline columns of
  Table 8 / Table 7 if they use the same field.

## Analysis and figures

### F4 — Fig 4/5 compare GBPR validation accuracy against baseline test accuracy — `open`
- **Evidence:** `experiments/gp_baseline/plot_progress.py:89-117` (with
  `# TODO: We need to store the test results`) uses `val_accs.npy` for GBPR.

### F16 — Fig 4 x-axis for GBPR is validation evaluations (every 10 epochs), not gradient steps — `open`
- **Evidence:** `plot_progress.py:120-147` indexes the per-job `val_accs.npy` series, which
  has one entry per 10 epochs (`VAL_STEP=10`). The caption says the x-axis counts gradient
  steps for GBPR.

### F18 — Fig 5 GBPR wall-clock time is estimated, not measured — `open`
- **Evidence:** GBPR time = (validation index) × `seconds_per_val_step` from
  `gbpr_timing.json`, a per-program average over a 10-job step-time benchmark
  (`train_mutations/benchmark_step_time.py`). GP/BFS time assumes a constant
  `duration_s / num_programs` per step (`plot_progress.py:168-190`).
- **Fix:** record wall-clock timestamps per validation step / per evaluated program.

### F10 — Repair threshold is defined inconsistently — `open`
- **Evidence:** `plot_progress.py:379` defaults `--threshold 1.0`;
  `analyze_test_results.py:822` uses `1 - linspace(0, 0.01, 5)`; `print_review_stats.py:30`
  and both dyck analyses hardcode 0.99; the LLM-patch evaluation used `== 1.0`.

### F11 — Maximum sequence length computed two ways — `open`
- **Evidence:** `gp_core.py:49` (max over train+val+test) vs
  `analyze_gp_bfs_dyck_balanced.py:~120` (test only).

### F12 — Fig 1 / Fig 6 are not reproducible from the repository — `open`
- **Evidence:** `trajectory.pkl` is gitignored (`experiments/train_mutations/.gitignore`);
  the job IDs behind the two figures are not recorded. Three local pickles exist (reverse
  `231a32a5…`, `66056586…`; sort `92e80c60…`) and are preserved in the backup.

### F20 — Several paper table cells were computed by hand — `open`
- **Evidence:** Table 2 "Repaired (≥99%)" (random) and "Epochs to early stop" (buggy) are
  produced by no script; Table 6 "Majority" column is computed by neither dyck script;
  Tables 6, 7, 8, 9 are printed to stdout and transcribed.

## Code health

### F13 — `test_gp_core.py` fails — `open`
- **Evidence:** `experiments/gp_baseline/test_gp_core.py:35` hardcodes 14 mutation sites;
  there are 16.

### F14 — Stale docstrings and help texts — `open`
- **Evidence:** `--random-init` help in `train_mutations.py:208-210` still says "compile the
  ground-truth program"; `plot_spec_size.py:13-16` and `print_review_stats.py:7-13` say
  1,135/1,466 jobs while the code uses 847.

### F15 — `Model.train()` crashes — `open`
- **Evidence:** `src/model.py:432` passes `returnAllMetrics=` to `Trainer`, which has no such
  parameter (`src/trainer.py:25-48`). Affects only legacy code paths (to be deleted).

### F21 — Editable install points at a stale path; imports rely on `sys.path` hacks — `open`
- **Evidence:** the `.venv` editable finder maps packages to `/home/andre/Repos/RASPing/…`;
  22 files append to `sys.path`; `src/functions.py:13` and
  `experiments/mutation/load_mutations.py` import each other (the latter keeps a copy of
  `getAcceptedNamesAndInput` to dodge the cycle).

### F22 — Unpinned core dependencies — `open`
- **Evidence:** `requirements.txt` pins `tracr` and `dm-haiku` to `@main`; `pyproject.toml`
  declares no dependencies; `uv.lock` contains only the project. The working `.venv` has
  tracr `9ce2b8c8`, haiku `47500fbd`.

### F23 — `jax_debug_nans=True` set globally on import — `open`
- **Evidence:** `src/model.py:14-16` changes global JAX config at import time (slows every
  computation and changes NaN behaviour for every importer).
