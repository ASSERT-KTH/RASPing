from pathlib import Path
import sys

# HACK: We need to fix the imports
module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from experiments.mutation.source.most_freq import make_sort_freq
from experiments.mutation.tests.test import Test


class TestMostFreq(Test):

    def setup_method(self, method):
        self.__setup__(make_sort_freq(self.maxLength), "most-freq")

    def test_model(self):
        self.test()
