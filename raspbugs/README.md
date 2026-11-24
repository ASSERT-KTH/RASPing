# RaspBugs: generating the benchmark

RaspBugs is a set of 847 buggy RASP programs ("mutants") obtained by mutating four reference
programs: hist (46), most_freq (284), reverse (268) and sort (249). The benchmark itself is
the manifest `raspbugs.jsonl.gz`: one record per mutant, holding its mutated source, diff,
operators and mutation order. Load it with `rasping.raspbugs`:

```python
from rasping import raspbugs
m = raspbugs.get("2a722d5d5449441fad93f98c73c633a2")   # or raspbugs.mutants(program="sort")
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
| `raspbugs.jsonl.gz` | the benchmark manifest: the 847 mutants |
| `architectures.py`, `architectures.csv` | each mutant's compiled architecture: parameter count, layers, shape signature (`rasping.raspbugs.shape_twins`) |

## Regenerating

Run both steps from this directory. Step 1 needs a fork of cosmic-ray with higher-order
mutations. It's installed by `uv sync`, pinned in `pyproject.toml`.

```bash
python run_mutations.py --orders 1 2 3 4 5     # writes results/<program>/ (gitignored)
python build_manifest.py                       # writes raspbugs.jsonl.gz, mutants.csv
```

The oracle labels each mutant as one of:

- `BUGGY_MODEL`: compiles, wrong on at least one oracle sample. These are the benchmark candidates.
- `CORRECT_MODEL`: an equivalent mutant.
- `UNCOMPILABLE_MODEL`: Tracr fails.
- `FAILED_MUTATION`: the mutated program errors.
- `TIMEOUT`
- `UNKNOWN_FAILURE`: no oracle output.

Up to 200 mutants are generated per program and order.

**Membership.** A mutant is in RaspBugs if the oracle labels it `BUGGY_MODEL` and its
compiled output vocabulary can represent the correct outputs (`representable`). On the raw
v1 outputs, the four programs give 1,118 buggy mutants, of which 271 are not representable.
The v1 paper (TMLR submission) also had two Dyck programs, for 1,466 mutants in total;
they were removed because their imbalanced binary outputs make exact-match accuracy
uninformative. See `docs/FINDINGS.md` F6, F7 and F35, including the 50-sample oracle used in v1.
