import unittest

from src.utils import get_file_path


class TestRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_hc.json")

    def test_runner(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass