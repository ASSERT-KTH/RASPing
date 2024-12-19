from tracr.rasp import rasp

from pathlib import Path
import sys

# HACK: We need to fix the imports
module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from experiments.mutation.source.reverse import make_reverse
from experiments.mutation.tests.test import Test


class TestReverse(Test):

    def setup_method(self, method):
        self.__setup__(make_reverse(rasp.tokens), "reverse")

    def test_model(self):
        self.test()
