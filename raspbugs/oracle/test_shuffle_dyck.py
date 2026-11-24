import source.shuffle_dyck as program_source
from oracle.test import Test

from rasping import programs
from rasping.config import MAX_LEN


class TestShuffleDyck(Test):
    def setup_method(self, method):
        self.__setup__(programs.get("shuffle_dyck").build(program_source, MAX_LEN), "shuffle_dyck")

    def test_model(self):
        self.test()
