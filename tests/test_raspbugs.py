"""The RaspBugs benchmark manifest and mutant compilation."""

import pytest

from rasping import raspbugs
from rasping.config import PROGRAMS
from rasping.model import Model


def test_benchmark_size():
    ms = raspbugs.mutants()
    assert len(ms) == 847 and len({m.job_id for m in ms}) == 847
    assert {p: len(raspbugs.mutants(program=p)) for p in PROGRAMS} == {
        "hist": 46, "most_freq": 284, "reverse": 268, "sort": 249}
    assert len(raspbugs.mutants(program="most-freq")) == 284
    assert {m.mutation_order for m in ms} == {1, 2, 3, 4, 5}


def test_mutant_fields():
    m = raspbugs.get("2a722d5d5449441fad93f98c73c633a2")
    assert (m.program, m.mutation_order) == ("hist", 1)
    assert m.operators == ("replace-rasp-comparison",)
    assert "make_hist" in m.source and m.diff


@pytest.mark.parametrize(
    "job_id",
    [
        "4795828b71974ec8829297c301a19dfe",  # sort
        "2b6f62a96b1d411a81185c47fcf323ab",  # reverse
        "2a722d5d5449441fad93f98c73c633a2",  # hist
        "d6de3b3af9fe403faeac2b7b7f0b5f85",  # most_freq
    ],
)
def test_compile_mutant(job_id):
    model = raspbugs.compile_mutant(job_id)
    assert isinstance(model, Model)
    assert model.name == raspbugs.get(job_id).program


@pytest.mark.slow
@pytest.mark.parametrize("program", PROGRAMS)
def test_benchmark_rule(program):
    """Every benchmark mutant compiles and can represent every target output (F6, F35)."""
    from rasping import data

    for m in raspbugs.mutants(program=program):
        model = m.compile()
        assert isinstance(model, Model)
        assert not data.unrepresentable_outputs(m.program, model), m.job_id
