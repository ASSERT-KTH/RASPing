from tracr.rasp import rasp

from pathlib import Path
import sys

# HACK: We need to fix the imports
module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from experiments.mutation.source.sort import make_sort
from experiments.mutation.tests.test import Test

from src.functions import getAcceptedNamesAndInput


class TestSort(Test):

    def setup_method(self, method):
        self.inputs = {t for t in getAcceptedNamesAndInput()["sort"]}
        self.__setup__(
            make_sort(
                rasp.tokens,
                rasp.tokens,
                max_seq_len=self.maxLength,
                min_key=min(self.inputs),
            ),
            "sort",
        )

    def test_model(self):
        self.test()
