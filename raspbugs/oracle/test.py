"""Mutation oracle: decides whether a mutated program is buggy.

cosmic-ray runs `pytest -s oracle/test_<program>.py` (from raspbugs/) on every mutant,
with the mutation applied to `source/<program>.py` in place. The printed markers are
parsed when the results are aggregated:
    "UNCOMPILABLE MODEL"   Tracr cannot compile the mutant
    "Accuracy: <x>"        accuracy of the compiled mutant on the first 50 test samples (F7)
    "MODEL IS NOT BUGGY"   the mutant is correct on those samples (the test then passes)
A failing assertion (accuracy < 1) means the mutant is buggy.
"""

import pytest

from rasping import data, programs
from rasping.config import MAX_LEN
from rasping.model import Model

N_ORACLE_SAMPLES = 50


class Test:
    def __setup__(self, sop, name):
        self.sop = sop
        self.name = programs.canonical(name)
        self.testing_data = data.load(self.name, "test")[:N_ORACLE_SAMPLES]

    @pytest.mark.skip(reason="This is not a test")
    def test(self):
        try:
            model = Model(self.sop, programs.get(self.name).vocab, MAX_LEN, self.name)
        except Exception as e:
            # Survive if the model is not compilable (we want to kill the buggy ones)
            print(str(e))
            print("UNCOMPILABLE MODEL")
            return
        accuracy = float(model.evaluate(self.testing_data).mean())
        print("Accuracy: ", accuracy)
        assert accuracy == 1.0
        print("MODEL IS NOT BUGGY")
