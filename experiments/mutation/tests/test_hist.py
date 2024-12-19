from pathlib import Path
import sys

# HACK: We need to fix the imports
module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from experiments.mutation.source.hist import make_hist
from experiments.mutation.tests.test import Test


class TestHist(Test):

    def setup_method(self, method):
        self.__setup__(make_hist(), "hist")

    def test_model(self):
        self.test()
