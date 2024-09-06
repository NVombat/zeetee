import unittest

from src.helper import get_file_path, get_key_by_value

class TestConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_hc.json")

    def test_get_key_by_value(self):
        pass

    def test_get_file_path(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass