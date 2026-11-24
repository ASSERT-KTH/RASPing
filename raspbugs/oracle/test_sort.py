import source.sort as program_source
from oracle.test import Test

from rasping import programs
from rasping.config import MAX_LEN


class TestSort(Test):
    def setup_method(self, method):
        self.__setup__(programs.get("sort").build(program_source, MAX_LEN), "sort")

    def test_model(self):
        self.test()
