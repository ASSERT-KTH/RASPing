# TracrBench mutants: the second benchmark

The second benchmark mutates the programs of **TracrBench** (Thurnherr & Riesen, 2024,
[arXiv:2409.13714](https://arxiv.org/abs/2409.13714),
[github.com/HannesThurnherr/TracrBench](https://github.com/HannesThurnherr/TracrBench)). It
uses the same mutation operators, oracle rule and benchmark rule as RaspBugs. TracrBench's
121 programs include 20 few-shot prompt examples without a ground truth. The other 101,
in `eval_data.json`, come with RASP code and a ground-truth Python function. They share
one input domain: tokens 0–9, length 2–10.

The TracrBench repository has no license. Its programs are copied here (`source/`,
`programs.json`) with attribution, at commit `48abdd9`.

```python
from rasping import tracrbench, programs
programs.tracrbench_programs()                  # the 77 included programs
m = tracrbench.mutants(program="sorting")[0]    # 1,153 mutants, 20 of them of sorting
model = m.compile()                             # mlp_exactness 1000
```

## Which programs are included

Rules 1–3 were fixed on Nov 25, 2026, before any mutant was generated. They are
`config.TracrBenchRules`. A program is included if all of these hold:

1. **It runs.** Its RASP code runs in TracrBench's own environment, where only `from tracr.rasp import rasp` is available.
2. **It compiles correctly.** Compiled with `mlp_exactness=1000` (TracrBench's setting), it reproduces the ground truth on every test example. Three programs that pass compile wrongly at RaspBugs' 120.
3. **It is not trivial.** Neither one constant output sequence nor copying the input reaches 50% exact-match accuracy on the test split.
4. **It is not a duplicate.** Programs with the same source up to names, comments and docstrings (`programs.canonical_source`) compile to the same circuit and have the same mutants. Only the first by name is kept.

Rule 4 was added the same day, after the mutants were generated but before the manifest was
built or any repair was run. The mutants showed count_frequency, count_occurrences and
histogram with identical counts in every order. The rule applies in step 0, and each
program's mutants are generated and sampled independently, so the result is the same as
applying it before generation. compute_median and set_to_median compute the same function
with different code (different circuits, different mutants), and both stay.

There is no size limit. A cap at 1.2M parameters (the largest RaspBugs model) was dropped
before any mutant was generated: it had no scientific basis, and it only excluded
element_divide (1.4M) and swap_min_max (2.8M). The largest program has 2.8M parameters
and the median 4.2k.

The result is **77 of 101 programs**: 36 use attention (Aggregate or SelectorWidth) and 41
are elementwise only. Report the two groups separately. An elementwise program is a per-token
lookup, so its mutants can only corrupt lookup tables.

| Excluded | Programs |
|---|---|
| does not run or compile (`math` not imported in TracrBench's environment) | arctangent, hyperbolic_cosine, hyperbolic_tangent, logarithm |
| ground truth fails on some inputs (math domain error) | arccosine, arcsine |
| reference program disagrees with the ground truth | check_power_of_n (51% of test examples), polynomial (0.2%) |
| trivial: copying the input | absolute, identity, invert_if_sorted, one_if_equal_to_next, round, sum_digits |
| trivial: a constant output | check_all_equal, check_alternating, check_descending, check_increasing, check_last_two_equal, check_multiple_of_n, check_palindrome, test_at_least_two_equal |
| duplicate of count_frequency | count_occurrences, histogram |

`programs.json` has every program's record: the reason, size, attention, trivial
baselines and ground truth.

## Specifications

`data/tracrbench/<name>_{train,val,test}.jsonl.gz` are generated like RaspBugs':
- 50,000 inputs drawn uniformly by length (2–10) and token, duplicates removed, seed 0;
- outputs from the ground truth, where a scalar result fills every position;
- a seeded 80/10/10 split.

## Pipeline

| Step | Command | Writes |
|---|---|---|
| 0 | `python tracrbench/import_programs.py` | `programs.json`, `source/`, `data/tracrbench/` (downloads `eval_data.json` at the pinned commit and checks its SHA-256) |
| 1 | `python tracrbench/launch_mutations.py --partition <cpu>` (one cluster job per program), or `python raspbugs/run_mutations.py --root tracrbench` locally | `results/<program>/` (gitignored) |
| 2 | `python raspbugs/build_manifest.py --root tracrbench` | `tracrbench.jsonl.gz`, `mutants.csv.gz` |
| 3 | `python raspbugs/architectures.py --benchmark tracrbench` | `architectures.csv` |

**Step 1.** It runs cosmic-ray with RaspBugs' operators: orders 1–5, up to 200 mutants per
program and order. The oracle `oracle/test_program.py` compiles each mutant and checks it
on the whole test split. While the model is compiled, it also checks that the output
vocabulary contains every target token.

**Step 2.** It labels the mutants and keeps the buggy, representable ones. It then selects
**at most 20 per program, stratified by mutation order** (seed 0; `mutants.csv.gz`, column
`selected`).

## The mutants

Generated on Berzelius on Nov 25, 2026 (SLURM array 18476981; 1–8 h per program on one CPU).

| | Mutants |
|---|---|
| generated (77 programs × orders 1–5) | 52,535 |
| `BUGGY_MODEL` | 40,179 |
| `UNCOMPILABLE_MODEL` | 7,078 |
| `UNKNOWN_FAILURE` (cosmic-ray outcome `no-test`: never run) | 2,475 |
| `CORRECT_MODEL` | 1,523 |
| `FAILED_MUTATION` | 651 |
| `TIMEOUT` (60 s; mostly count_prime_factors and next_prime) | 629 |
| buggy and representable | 17,283 |
| **selected: the benchmark** | **1,153** |

**58 of the 77 programs contribute mutants**: 56 have 20, multiply_by_length has 19 and
next_prime has 14. Some programs have no representable mutant at some orders, so their
orders aren't balanced (`mutants.csv.gz`).

**19 programs contribute none**: cosine, cube_each_element, cube_root, decrement,
element_double, element_subtract_constant, exponential, factorial, hyperbolic_sine, increment,
increment_by_index, negation, position, set_to_index, sine, square_each_element, square_root,
tangent and triple. All are elementwise numeric maps. Every buggy mutant of them is
unrepresentable. A mutated map such as `x + 1` → `x - 1` compiles to an output vocabulary
without some correct outputs (here 9 and 10), so no weights can repair it. They stay in the
program list, which is what the rules give, and the benchmark reports them as
contributing no mutants.

count_frequency, count_greater_than and count_less_than differ only in the comparison
(`EQ`, `GT`, `LT`), so they have the same mutation sites and the same counts per order.
They are different functions with different specifications, so their mutants are different
repair tasks.

The selected mutants' compiled accuracy (exact match on the whole test split, the
manifest field `oracle_accuracy_50`, named after RaspBugs' 50-sample oracle) has median
0.008. 12% are at 0 and 3% above 0.5. `architectures.csv` has each mutant's parameter count
and shape signature: median 18.8k parameters, maximum 3.4M (a swap_min_max mutant).
