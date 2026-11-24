"""The RaspBugs benchmark manifest and mutant compilation."""

import pytest

from rasping import raspbugs
from rasping.config import MAIN_PROGRAMS, PROGRAMS
from rasping.model import Model


def test_subset_sizes_v1():
    assert len(raspbugs.mutants("buggy")) == 1773
    assert len(raspbugs.mutants("all")) == 1466
    assert len(raspbugs.mutants("main")) == 847
    assert len(raspbugs.mutants("dyck")) == 619
    assert {m.program for m in raspbugs.mutants("main")} == set(MAIN_PROGRAMS)
    assert len(raspbugs.mutants("all", program="most-freq")) == 284


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
        "28357cf95fbd4389a311073d185756b8",  # shuffle_dyck
        "95b38e6e8ede42268ff62e85f987dfa7",  # shuffle_dyck2
    ],
)
def test_compile_mutant(job_id):
    model = raspbugs.compile_mutant(job_id)
    assert isinstance(model, Model)
    assert model.name == raspbugs.get(job_id).program


@pytest.mark.slow
@pytest.mark.parametrize("program", PROGRAMS)
def test_compile_all_mutants(program):
    for m in raspbugs.mutants("all", program=program):
        assert isinstance(m.compile(), Model)
