import source.shuffle_dyck2 as program_source
from oracle.test import Test

from rasping import programs
from rasping.config import MAX_LEN


class TestShuffleDyck2(Test):
    def setup_method(self, method):
        self.__setup__(programs.get("shuffle_dyck2").build(program_source, MAX_LEN), "shuffle_dyck2")

    def test_model(self):
        self.test()
