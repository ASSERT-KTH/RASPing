"""Mutation oracle for TracrBench programs: decides whether a mutated program is buggy.

cosmic-ray runs `pytest -s oracle/test_program.py::test_model[<program>]` (from the
program's working copy, see raspbugs/run_mutations.py) with the mutation applied to
`source/<program>.py`. The printed markers are those of the RaspBugs oracle
(raspbugs/oracle/test.py), parsed by build_manifest.py:
    "UNCOMPILABLE MODEL"   Tracr cannot compile the mutant
    "REPRESENTABLE: <b>"   whether the compiled output vocabulary contains every target token
                           of the specification (the benchmark rule, `data.unrepresentable_outputs`)
    "Accuracy: <x>"        exact-match accuracy of the compiled mutant on the whole test split
    "MODEL IS NOT BUGGY"   the mutant is correct on every test example (the test passes)
Building the mutated program happens in a fixture, so a mutation that breaks the program
(TypeError, ...) is reported in pytest's ERRORS section, as for RaspBugs.
"""

from pathlib import Path

import pytest

from rasping import data, programs
from rasping.config import MAX_LEN
from rasping.model import Model

SOURCE = Path(__file__).resolve().parents[1] / "source"
NAMES = [p.name for p in programs.tracrbench_programs()]


@pytest.fixture(params=NAMES, ids=NAMES)
def name(request):
    return request.param


@pytest.fixture
def sop(name):
    return programs.build_sop((SOURCE / f"{name}.py").read_text(), name, MAX_LEN)


def test_model(name, sop):
    p = programs.get(name)
    try:
        model = Model(sop, p.vocab, MAX_LEN, name, mlp_exactness=p.mlp_exactness)
    except Exception as e:
        # Survive if the model is not compilable (we want to kill the buggy ones)
        print(str(e))
        print("UNCOMPILABLE MODEL")
        return
    print("REPRESENTABLE: ", not data.unrepresentable_outputs(name, model))
    accuracy = float(model.exact_match(data.load(name, "test")).mean())
    print("Accuracy: ", accuracy)
    assert accuracy == 1.0
    print("MODEL IS NOT BUGGY")
