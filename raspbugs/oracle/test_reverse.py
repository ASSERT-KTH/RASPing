import source.reverse as program_source
from oracle.test import Test

from rasping import programs
from rasping.config import MAX_LEN


class TestReverse(Test):
    def setup_method(self, method):
        self.__setup__(programs.get("reverse").build(program_source, MAX_LEN), "reverse")

    def test_model(self):
        self.test()
