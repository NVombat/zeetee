import unittest

from src.utils import get_file_path
from src.helper import (
    json_to_rgp,
    rgp_dict_to_rgp,
    extract_clauses,
    get_key_by_value,
    generate_unique_pairs,
    json_to_dict,
    get_constraint_intersection
)

class TestHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_hc.json")

    def test_json_to_rgp(self):
        pass

    def test_rgp_dict_to_rgp(self):
        pass

    def test_extract_clauses(self):
        pass

    def test_get_key_by_value(self):
        pass

    def test_generate_unique_pairs(self):
        pass

    def test_json_to_dict(self):
        pass

    def test_get_constraint_intersection(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass