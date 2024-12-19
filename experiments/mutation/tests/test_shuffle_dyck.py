from pathlib import Path
import sys

# HACK: We need to fix the imports
module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from experiments.mutation.source.shuffle_dyck import make_shuffle_dyck
from experiments.mutation.tests.test import Test


class TestShuffleDyck(Test):

    def setup_method(self, method):
        self.__setup__(make_shuffle_dyck(["()"]), "shuffle_dyck1")

    def test_model(self):
        self.test()
