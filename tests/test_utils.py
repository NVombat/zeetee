import unittest

from src.utils import get_file_path

class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_hc.json")

    def test_get_file_path(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass