# RaspBugs: generating the benchmark

RaspBugs is a set of buggy RASP programs ("mutants") obtained by mutating six reference
programs. The benchmark itself is the manifest `raspbugs_<version>.jsonl.gz`: one record
per buggy mutant, holding its mutated source, diff, operators and mutation order. Load it
with `rasping.raspbugs`:

```python
from rasping import raspbugs
m = raspbugs.get("2a722d5d5449441fad93f98c73c633a2")   # or raspbugs.mutants("main")
model = m.compile()                                    # Tracr -> Transformer
```

```
python -m rasping.raspbugs <job_id>     # print a mutant and its diff
```

## Layout

| Path | What |
|---|---|
| `source/<program>.py` | reference RASP programs (from Weiss et al. 2021 / Tracr's `lib`) |
| `operators/` | the 9 RASP-specific cosmic-ray mutation operators (package `rasp_operators`); the other 6 operators are cosmic-ray's built-in ones |
| `oracle/` | the test cosmic-ray runs on each mutant to label it (compiles it, measures accuracy on 50 test samples) |
| `run_mutations.py` | step 1: write cosmic-ray configs and run them per program and mutation order |
| `build_manifest.py` | step 2: re-apply diffs, label mutants from the oracle output, write the manifest |
| `raspbugs_v1.jsonl.gz` | the v1 (TMLR submission) manifest: 1,773 buggy mutants, 1,466 flagged `in_raspbugs_v1` |

## Regenerating

Run both steps from this directory. Step 1 needs a fork of cosmic-ray with higher-order
mutations. It's installed by `uv sync`, pinned in `pyproject.toml`.

```bash
python run_mutations.py --orders 1 2 3 4 5     # writes results/<program>/ (gitignored)
python build_manifest.py --version v2          # writes raspbugs_v2.jsonl.gz, mutants_v2.csv
```

The oracle labels each mutant as one of:

- `BUGGY_MODEL`: compiles, wrong on at least one oracle sample. These are the benchmark candidates.
- `CORRECT_MODEL`: an equivalent mutant.
- `UNCOMPILABLE_MODEL`: Tracr fails.
- `FAILED_MUTATION`: the mutated program errors.
- `TIMEOUT`
- `UNKNOWN_FAILURE`: no oracle output.

Up to 200 mutants are generated per program and order.

**v1 membership.** The v1 benchmark (1,466) is the set of buggy mutants whose compiled
output vocabulary can represent the correct outputs. That filter arose implicitly: in v1,
training crashed on the other 307 mutants. See `docs/FINDINGS.md` F6 and F7 for this and
for the 50-sample oracle. To rebuild v1 exactly from the archived raw outputs:

```bash
python build_manifest.py --version v1 --results-dir <archive>/experiments/mutation/results \
    --membership-from <csv with the 1,466 job_ids>
```
