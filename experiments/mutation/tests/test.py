from tracr.rasp import rasp

from pathlib import Path
from abc import ABC
import pytest
import sys

# HACK: We need to fix the imports
module_paths = [
    str(Path(Path(__file__).parent.resolve(), "..", "..", "..").resolve().absolute())
]
if module_paths not in sys.path:
    sys.path.extend(module_paths)

from src.functions import getAcceptedNamesAndInput, load_dataset
from src.model import Model


class Test(ABC):

    maxLength = 10

    def __setup__(self, model, name):
        self.model = model
        self.name = name
        self.inputs = {t for t in getAcceptedNamesAndInput()[self.name]}
        # Load test data instead of generating it
        data_dir = Path(__file__).parent.resolve().parent.parent.parent / "data"
        self.testing_data = load_dataset(data_dir, self.name, "test")[:50]

    @pytest.mark.skip(reason="This is not a test")
    def test(self):
        try:
            model = Model(self.model, self.inputs, self.maxLength, self.name)
        except Exception as e:
            # Survive if the model is not compilable (we want to kill the buggy ones)
            print(str(e))
            # Print message for post-processing
            print("UNCOMPILABLE MODEL")
            return

        accuracy = model.evaluateModel(
            self.testing_data, doPrint=False, outputArray=False, useAssert=False
        )
        # Print the accuracy for post-processing
        print("Accuracy: ", accuracy)
        assert accuracy == 1.0
        # Survive if the model is not buggy, print message for post-processing
        print("MODEL IS NOT BUGGY")
