import source.most_freq as program_source
from oracle.test import Test

from rasping import programs
from rasping.config import MAX_LEN


class TestMostFreq(Test):
    def setup_method(self, method):
        self.__setup__(programs.get("most_freq").build(program_source, MAX_LEN), "most_freq")

    def test_model(self):
        self.test()
