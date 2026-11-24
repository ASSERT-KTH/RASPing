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

Raw v1 data (run directories with weights, raw cosmic-ray outputs, trajectories) was
archived on Nov 24, 2026 as `worktree.tar.zst` (24.7 GB raw, 3.4 GB compressed) together
with a full mirror of the pre-rewrite git history; see the README for where it lives.

---

## Summary: what this means for the resubmission

**Reruns needed (results in `results/v1_tmlr/` no longer match the code):**

| Results | Why | Findings |
|---|---|---|
| All GP / BFS runs (Figs 4–5, Table 7, Table 8 baseline columns) | budget-only BFS, train-only fitness, no test tie-breaks, failures counted, `MAX_LEN`, timestamps | F3, F5, F11, F18, F29, F30, F31 |
| Dyck-2 GBPR and baseline runs (App. C) | new Dyck-2 specification (lengths 2–10) | F1 |
| Spec-size ablation (Table 8, Fig 8) | N=40,000 inside the 500k-step budget; 1,000-program baselines | F8 |
| Main GBPR runs (Table 1/5, Figs 3–5, 7) | not invalidated numerically; rerun to get selected-test curves, measured times and full-test "before" accuracies for Figs 3–5/7 | F4, F7, F16, F18 |
| Dyck metrics (Tables 6–7) | recall definition | F28 |

**Paper text to change regardless of reruns:** early-stopping patience is 50 (F2); state
the benchmark rule — buggy mutants whose compiled output vocabulary can represent the
correct outputs — and that it favours GBPR over GP/BFS (F6); oracle accuracy was on 50
samples (F7); list all six mutant statuses (F19); Fig 4's GBPR axis is evaluations every
10 epochs, not gradient steps (F16); §5.3 quotes test-based GBPR rates next to a
validation-based curve (F4); Table 8's bold entry at order 4, N=10k is wrong (F27).

**Not affected:** every number in the submitted paper was reproduced exactly from the raw
runs (`tests/test_paper_numbers.py`); the refactored code reproduces the v1 code's compiled
weights and training numerics (`tests/test_golden.py`).

---

## Data and benchmark

### F1 — Dyck-2 specification never contains inputs longer than 8 — `fixed`
- **Evidence:** `src/functions.py:71` — `generateData(exhaustive=True)` enumerates inputs in
  order of length and returns `data[:size]` *before* shuffling. For `shuffle_dyck2` with
  n=50,000 and max_len=10, every input in `data/shuffle_dyck2_*.jsonl` has length 2–8.
- **Paper:** §4 says input lengths are sampled between 2 and 10.
- **Invalidates:** all `shuffle_dyck2` results (App. C Tables 6–7; spec-size Dyck runs).
- **Fix (see commit "Fix specification generation"):** exhaustive enumerations larger than
  the requested size are sampled with a seeded, length-stratified draw
  (`rasping/data.py::_sample_by_length`); `data/shuffle_dyck2_*.jsonl` regenerated (seed 0):
  lengths 2–10 now present (train: 12/52/219/806/3349 at lengths 2–6, ~8.9k each at 7–10).
  Balanced strings remain rare (0.5%), as before. The golden `shuffle_dyck2` training entry
  was re-recorded. Mutant membership is unaffected (labels depend only on the program).

### F6 — The 1,466-mutant benchmark is defined by a join with GBPR results — `fixed` (rule made explicit)
- **Evidence:** `experiments/mutation/raspbugs.ipynb` inner-joins
  `aggregated_mutations.json` (1,773 `BUGGY_MODEL` rows) with
  `train_mutations/analysis_outputs/all_test_accuracies.csv` (GBPR results) to produce
  `raspbugs.json`. 307 buggy mutants are dropped; ~100 are explained by duplicate sources,
  the rest are undocumented and skew toward reverse/sort/hist at orders 3–5. Every
  experiment script loads `aggregated_mutations.json` instead and re-filters (or doesn't).
- **Root cause (verified Nov 24, 2026):** v1 trained *all* 1,773 buggy mutants; 306 of the 307
  without a result have a compiled **output vocabulary that cannot represent the correct
  outputs** (hist 48, most_freq 62, reverse 84, sort 77, shuffle_dyck 14, shuffle_dyck2 21),
  so encoding the specification raised and the training job crashed. Every one of the 1,466
  benchmark mutants is representable. The remaining mutant (shuffle_dyck2, representable)
  failed for an unknown reason. So v1 RaspBugs = "buggy mutants whose compiled architecture
  can represent the target" — a sensible but undocumented filter that only applies to GBPR:
  GP/BFS could repair the excluded mutants, so excluding them from the comparison favours GBPR.
- **Invalidates:** benchmark size and composition (§4, every table); the GBPR-vs-symbolic
  comparison (§5.3) should either include the 306 (GBPR counted as failing) or state the filter.
- **Decision (Nov 24, 2026):** keep the v1 set of 1,466 mutants, with the rule stated explicitly:
  *buggy mutants whose compiled output vocabulary can represent the specification's outputs*.
  Sound because the baselines are evaluated on the same set. The paper must state the rule
  and that it excludes mutants GP/BFS might repair.
- **Fix:** `data.unrepresentable_outputs()` implements the rule; `tests/test_raspbugs.py::
  test_benchmark_rule_v1` (slow) checks that it reproduces v1 membership (one Dyck-2
  exception); `build_manifest.py` applies it for new benchmark versions; `rasping.repair`
  records unrepresentable mutants as `status: unrepresentable` instead of crashing.

### F7 — "Buggy" label decided on only 50 test samples — `fixed` (pipeline)
- **Evidence:** `experiments/mutation/tests/test.py:29` loads `data/<prog>_test.jsonl[:50]`.
  The pre-repair accuracy stored in `execution_result.accuracy` (and plotted as "Before GBPR"
  in Figs 3 and 7) is also computed on these 50 samples, whereas post-repair accuracy is on
  the full 5,000-sample test split.
- **Paper:** §4 says a buggy program must be incorrect on the input-output specification.
- **Invalidates:** benchmark membership (mutants correct on the 50 but wrong elsewhere are
  mislabelled `CORRECT_MODEL`); the "before" histograms and the "min/median/mean" statistics
  in §4.
- **Fix:** the oracle uses the whole test split for future benchmark builds; repair runs
  record the compiled mutant's accuracy on the full test split (`initial_test_accuracy`),
  which Figs 3/7 use for v2. v1 membership is kept (decision Nov 24, 2026): a mutant wrong
  on one of the 50 samples is buggy on the full specification too, so no v1 mutant is
  mislabelled; only bugs invisible on those 50 samples may be missing.

### F19 — Undocumented mutant statuses — `fixed`
- **Evidence:** `aggregated_mutations.json` has, besides the four statuses described in §4,
  `UNKNOWN_FAILURE` (362) and `TIMEOUT` (78).
- **Fix:** documented in `raspbugs/README.md` (UNKNOWN_FAILURE = no oracle output); the paper
  should list all six.

## Training (GBPR)

### F2 — Early-stopping patience: paper says 10, code used 50 — `paper`
- **Evidence:** `experiments/train_mutations/train_mutations.py:170` CLI default
  `early_stopping_patience=50`; no launcher overrides it (the function default of 10 at :37
  is never reached). `launch_random_init_baseline.py:11-12` confirms 50.
- **Paper:** §5.1 "validation loss improvement < 1e-4 for 10 epochs".
- **Decision (Nov 24, 2026):** keep 50 (`config.TrainConfig`); correct the paper text to 50.

### F8 — Spec-size ablation: N=40,000 point and baselines don't match the stated setup — `fixed`
- **Evidence:** `experiments/spec_size/launch_spec_size.py:8` runs N ∈ {100,…,25,000} with
  `N_STEPS=500k`; the N=40,000 column reuses the main runs, which have `n_epochs=10000`
  (≈1.56M step cap). The GP/BFS columns use the 500-program budget runs while the main
  experiments use 1,000.
- **Paper:** App. D says the gradient-step budget is fixed at 500k for all N.
- **Fix:** `experiments/spec_size/launch.py` includes N=40,000 under the 500k budget; the
  analysis uses the main 1,000-program baseline runs for v2 (v1 keeps 500 to reproduce).

## Symbolic baselines (GP / BFS)

### F3 — BFS is told the true mutation order — `fixed`
- **Evidence:** `experiments/gp_baseline/run_gp_exhaustive.py:330` sets
  `max_depth = mutation_order` of the mutant being repaired.
- **Paper:** not disclosed; this is oracle information on top of the operator oracle §5.3
  already acknowledges.
- **Fix (commit "Fix the symbolic baselines"):** BFS has no depth limit by default (budget only), decided
  Nov 24, 2026. `--max-depth` remains available.

### F5 — GP fitness uses train+val; paper says train — `fixed`
- **Evidence:** `experiments/gp_baseline/gp_core.py:375` (BFS likewise).
- **Fix (commit "Fix the symbolic baselines"):** GP and BFS fitness = training-split accuracy.

### F9 — Conflicting GP defaults; GP and BFS share an output dir — `fixed`
- **Evidence:** CLI defaults pop=16, k=3, budget=500 (`run_gp_baseline.py`) vs
  `run_gp_for_bug` defaults pop=60, k=4 (`gp_core.py:675-677`). Both runners default
  `--output-dir` to `experiments/gp_baseline/results` (`run_gp_baseline.py:120`,
  `run_gp_exhaustive.py:392`).
- **Fix (228d70d):** one set of defaults (`gp.DEFAULTS`, the paper's), output dirs
  `runs/symbolic_baselines/<method>_budget<B>/`.

### F17 — Baseline progress curves use the running maximum of the selected program's test accuracy — `fixed`
- **Evidence:** `best_test_acc` in the search histories is the test accuracy of the
  currently *selected* (best-fitness) program — not of any candidate, as first written
  here. But the curves (`plot_progress.py` `expand_gp_history_to_series`) and Table 8
  (`max_best_test_acc`) take its running maximum over time, so a mutant counts as repaired
  if the selection was ever ≥99% on test, even if later replaced.
- **Impact on v1:** none at the 99% threshold — running maximum and final selection give
  identical pass rates for every method, budget and mutation order (checked Nov 24, 2026).
- **Fix:** with F29 fixed the selected program's test accuracy is non-decreasing in
  fitness order only by chance, so v2 analysis should use the selected program's value at
  each step (`test_acc` of the final record for endpoints). See F4 for the GBPR side.

## Analysis and figures

### F4 — Fig 4/5 compare GBPR validation accuracy against baseline test accuracy — `fixed` (logging)
- **Evidence:** `experiments/gp_baseline/plot_progress.py:89-117` (with
  `# TODO: We need to store the test results`) uses `val_accs.npy` for GBPR. The §5.3 text then
  quotes the *test*-based rates (41.9% / 26.0% at orders 4 / 5, = Table 5) while the plotted
  GBPR plateaus are the validation-based 41.2% / 24.7%.
- **Fix (commit "Log the selected parameters' test accuracy"):** the Trainer records, at every
  validation step, the test accuracy of the parameters early stopping would return then
  (best validation loss so far): `selected_test_accs` in curves.npz / `selected_test_acc` in
  `val_curves.csv.gz`. `experiments/symbolic_baselines/analyze.py` uses it (and the baselines'
  selected-program test accuracy without running max, F17) when present; v1 tables still
  reproduce the submitted figures exactly.

### F16 — Fig 4 x-axis for GBPR is validation evaluations (every 10 epochs), not gradient steps — `fixed`
- **Evidence:** `plot_progress.py:120-147` indexes the per-job `val_accs.npy` series, which
  has one entry per 10 epochs (`VAL_STEP=10`). The caption says the x-axis counts gradient
  steps for GBPR.
- **Fix:** the axis is labelled for what it is (GBPR: evaluations every 10 epochs; GP/BFS:
  programs evaluated); the caption must say so. Fig 5 (time) is the comparable view.

### F18 — Fig 5 wall-clock time is estimated, not measured — `fixed` (logging)
- **Evidence:** GBPR time = (validation index) × `seconds_per_val_step` from
  `gbpr_timing.json`, a per-program average over a 10-job step-time benchmark
  (`train_mutations/benchmark_step_time.py`). GP/BFS time assumes a constant
  `duration_s / num_programs` per step (`plot_progress.py:168-190`).
- **Fix:** GBPR runs record per-epoch timestamps (`run.json` → `curves.csv.gz:time_s`, 228d70d);
  GP/BFS histories record `time_s` per evaluated program. The v2 analysis must use them.

### F10 — Repair threshold is defined inconsistently — `fixed`
- **Evidence:** `plot_progress.py:379` defaults `--threshold 1.0`;
  `analyze_test_results.py:822` uses `1 - linspace(0, 0.01, 5)`; `print_review_stats.py:30`
  and both dyck analyses hardcode 0.99; the LLM-patch evaluation used `== 1.0`.
- **Fix (228d70d):** `config.THRESHOLD` is the only definition.

### F11 — Maximum sequence length computed two ways — `fixed`
- **Evidence:** `gp_core.py:49` (max over train+val+test) vs
  `analyze_gp_bfs_dyck_balanced.py:~120` (test only); both include BOS, so candidates were
  built with max_len 11 while compiled models use 10.
- **Fix (commit "Fix the symbolic baselines"):** searches build candidates with `config.MAX_LEN`.

### F12 — Fig 1 / Fig 6 are not reproducible from the repository — `fixed`
- **Evidence:** `trajectory.pkl` is gitignored (`experiments/train_mutations/.gitignore`);
  the job IDs behind the two figures are not recorded. Three local pickles exist (reverse
  `231a32a5…`, `66056586…`; sort `92e80c60…`) and are preserved in the backup.
- **Fix (228d70d):** identified by parameter/checkpoint counts: Fig 1 = reverse
  `231a32a5efd24a10840857bd72e478a1`, Fig 6 = sort `92e80c60de964d53b02dc7e968a0559e`;
  recorded in `experiments/trajectories/{launch,plot}.py`; `launch.py` re-runs exactly these.

### F20 — Several paper table cells were computed by hand — `fixed`
- **Evidence:** Table 2 "Repaired (≥99%)" (random) and "Epochs to early stop" (buggy) are
  produced by no script; Table 6 "Majority" column is computed by neither dyck script;
  Tables 6, 7, 8, 9 are printed to stdout and transcribed.
- **Fix (228d70d):** every table is written as `.tex` by `experiments/*/analyze.py`
  (`scripts/reproduce_paper.py`); the Majority column is computed from the test splits.

## Code health

### F13 — `test_gp_core.py` fails — `fixed`
- **Evidence:** `experiments/gp_baseline/test_gp_core.py:35` hardcodes 14 mutation sites;
  there are 16.
- **Fix (228d70d):** `tests/test_baselines.py` asserts the per-operator counts (total 16).

### F14 — Stale docstrings and help texts — `fixed`
- **Evidence:** `--random-init` help in `train_mutations.py:208-210` still says "compile the
  ground-truth program"; `plot_spec_size.py:13-16` and `print_review_stats.py:7-13` say
  1,135/1,466 jobs while the code uses 847.
- **Fix (228d70d):** the scripts were rewritten (`rasping/repair.py`, `experiments/spec_size/`).

### F15 — `Model.train()` crashes — `fixed`
- **Evidence:** `src/model.py:432` passes `returnAllMetrics=` to `Trainer`, which has no such
  parameter (`src/trainer.py:25-48`). Affects only legacy code paths (to be deleted).
- **Fix (cd73c9a):** removed with the legacy code paths.

### F21 — Editable install points at a stale path; imports rely on `sys.path` hacks — `fixed`
- **Evidence:** the `.venv` editable finder maps packages to `/home/andre/Repos/RASPing/…`;
  22 files append to `sys.path`; `src/functions.py:13` and
  `experiments/mutation/load_mutations.py` import each other (the latter keeps a copy of
  `getAcceptedNamesAndInput` to dodge the cycle).
- **Fix (cd73c9a, 228d70d):** `rasping` package installed by `uv sync`; no `sys.path` hacks;
  scripts run from any directory.

### F22 — Unpinned core dependencies — `fixed`
- **Evidence:** `requirements.txt` pins `tracr` and `dm-haiku` to `@main`; `pyproject.toml`
  declares no dependencies; `uv.lock` contains only the project. The working `.venv` has
  tracr `9ce2b8c8`, haiku `47500fbd`.
- **Fix (cd73c9a):** all deps in `pyproject.toml`, git deps pinned to commits, transitive deps
  constrained to the v1 environment, real `uv.lock`.

### F23 — `jax_debug_nans=True` set globally on import — `fixed`
- **Evidence:** `src/model.py:14-16` changes global JAX config at import time (slows every
  computation and changes NaN behaviour for every importer).
- **Fix:** removed; the Trainer stops when the epoch loss is non-finite, restores the last
  finite parameters and records `diverged` in run.json. Golden tests unchanged.

## Found during the refactor (Nov 24, 2026)

### F24 — Specifications are not reproducible (unseeded RNG) — `fixed`
- **Evidence:** the random generators in `rasping/generators.py` draw from NumPy's global RNG
  and `scripts/generate_data.py` never seeds it, so `data/` cannot be regenerated.
- **Fix:** `data.generate(..., seed=)` and `generate_data.py --seed` (default 0). The committed
  v1 files of hist/most_freq/reverse/sort are kept as is (all v1 results use them); they
  cannot be byte-reproduced, only regenerated with the same distribution.

### F25 — `most_freq` exhaustive generator breaks ties differently — `fixed`
- **Evidence:** `generate_most_freq_exhaustive` sorts by `(-count, token)`; the random
  generator (used for `data/`) and Tracr's `make_sort_freq` keep first-occurrence order.
  Only the exhaustive generator (tests) is affected.
- **Fix:** stable sort by count, as Tracr; test updated.

### F26 — Training-time trajectory storage on every main run — `fixed`
- **Evidence:** v1 `launch_train_mutations.py` ran every mutant with `store_trajectory=True`
  (full parameter copy every 50 steps, kept in memory until the end). Long runs of large
  models (most_freq: up to 1.1M parameters) can exhaust memory. (It is *not* the cause of
  the missing results in F6.) `experiments/main_repair/launch.py` no longer stores
  trajectories.

### F27 — Table 8 bold entry contradicts its caption — `paper`
- **Evidence:** order 4, N=10k (18.4) is bold ("GBPR surpasses both GP and BFS") but BFS is
  20.6. `experiments/spec_size/analyze.py` bolds by the caption's rule.

### F28 — Dyck token recall ignores failed predictions — `fixed`
- **Evidence:** `rasping/metrics.py::dyck_metrics` (v1 definition) counts a false negative only
  when the prediction is 0; positions where a GP/BFS program crashed (label −1) count as
  neither TP nor FN, inflating token recall / balanced accuracy for those programs (Table 6
  columns for baselines; Table 7 uses sequence-level metrics and is unaffected).
- **Fix:** failed predictions on positive tokens count as false negatives. The v1 tables in
  `results/v1_tmlr/dyck/` were computed with the old definition (frozen).

### F29 — Symbolic search breaks fitness ties with test accuracy — `fixed`
- **Evidence:** `rasping/baselines/gp.py:145,230`, `bfs.py:116`: among candidates with equal
  fitness the one with higher *test* accuracy is kept as best, so `best_source`/`test_acc`
  depend on the test set (separate from F17, which concerns the progress curves).
- **Fix (commit "Fix the symbolic baselines"):** ties keep the incumbent (first found); test accuracy is
  computed for reporting only.

### F30 — Compile/evaluation failures are silently scored 0 — `fixed`
- **Evidence:** `rasping/baselines/common.py:135-143` returns 0.0 on any error or timeout;
  `num_compile_failures` is therefore always 0 and broken candidates are recorded as `ok`.
- **Fix (commit "Fix the symbolic baselines"):** failures raise `CandidateError`; they are counted
  (`num_compile_failures`, `num_timeouts`), logged as `compile_fail` and excluded from the
  population/frontier. (On the golden mini-runs GP had 5 and 2 such failures.)

### F31 — Minor inconsistencies in the symbolic baselines — `fixed` (partly)
- GP status is `success_early` only if the perfect program is in the initial population;
  generation 2 never appears (double increment, `gp.py:171,196`); survivors are not
  tie-broken as the comment says (`gp.py:243`); GP mutations are not checked for no-ops
  while BFS drops them; BFS `duration_s` excludes data loading, GP's includes it;
  candidates are built with `max_len=11` (BOS counted) while compiled models use 10 (F11);
  duplicates in the initial population were counted twice.
- **Fix (commit "Fix the symbolic baselines"):** status `success_early` whenever a perfect-fitness program
  is found; consecutive generation numbers; `duration_s` excludes data loading for both;
  duplicates counted once. GP still does not filter no-op mutations (`wontfix`: part of the
  algorithm as published).
- The v1 spec-size baseline tables (budget 500) contain 1,691 GP / 1,760 BFS records — an
  older, larger mutant set; the analysis restricts them to the benchmark.

### F32 — GP/BFS fork worker processes after JAX is initialised — `fixed`
- **Evidence:** `evaluate_with_timeout` uses `multiprocessing` with the default fork start
  method; Python warns this can deadlock with JAX's threads.
- **Fix:** JAX is now imported only by the modules that use it (`model`, `losses`, `train`;
  lazily in `data.encode` / `metrics.exact_match_accuracy`), so the symbolic baselines
  never load JAX and forking is safe. Golden search results unchanged.

### F33 — Oracle accuracy round-tripped through pandas — `wontfix`
- 80 `oracle_accuracy_50` values in `raspbugs_v1.jsonl.gz` carry float noise (0.9400000000000001)
  from the v1 pandas JSON round trip; `build_manifest.py` writes the parsed value. Harmless.

### F34 — Golden training numerics are only bit-exact on the recording machine — `fixed` (test)
- **Evidence:** CI (GitHub runner, 4 vCPUs) failed `test_golden_training` for most_freq, sort
  and shuffle_dyck. Pinning this machine to 4 or 1 cores (`taskset`) fails it too. Compiled
  weights, encodings and accuracies match everywhere. Training losses agree closely in the
  first epoch (to ~1e-12 locally, 2e-6 relative on CI), then diverge: by epoch 4, most_freq is
  off by 2.1%. Cause: XLA's CPU backend splits
  reductions by thread count and ISA, so the summation order changes. The compiled models'
  sharp logits and Adam amplify those last-bit differences within a few steps.
- **Consequence:** the "new code = v1 code to 1e-9" claim holds on the recording machine
  (16 threads), where it was established. The same holds for any v1-vs-v2 training comparison:
  differences in late-epoch numbers between machines are expected and are not bugs. It is no
  concern for the paper's results, which are pass rates at the 0.99 threshold over many mutants.
- **Fix (test only):** by default `tests/test_golden.py` checks weights and accuracies exactly,
  the first training epoch to 2e-5 relative, and later losses to 10%. A 1% learning-rate change
  moves the first-epoch losses by at least 2.5e-4 for every program, so it still fails. `pytest --golden-exact` restores the full 1e-7 comparison, including
  the trained weights, for use on the recording machine. Golden values unchanged.
- **Related:** the CI log warns about `os.fork()` with JAX during `test_golden_search`. That only
  happens because other tests in the same pytest process have imported JAX. The baselines on
  their own do not load JAX (F32).
