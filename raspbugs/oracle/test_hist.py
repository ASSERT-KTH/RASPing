import source.hist as program_source
from oracle.test import Test

from rasping import programs
from rasping.config import MAX_LEN


class TestHist(Test):
    def setup_method(self, method):
        self.__setup__(programs.get("hist").build(program_source, MAX_LEN), "hist")

    def test_model(self):
        self.test()
