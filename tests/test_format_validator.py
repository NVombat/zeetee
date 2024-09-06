import unittest

from src.helper import get_file_path
from src.format_validator import validate_json_format
class TestFormatValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_hc.json")
        json_file_path = get_file_path("testfiles", "rgp_false_fmt.json")

    def test_validate_json_format(self):
        pass

    def test_invalidate_json_format(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass