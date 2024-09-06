import unittest

from src.helper import get_file_path
from src.er_encoder import rgp_to_sat_er
from src.mb_encoder import rgp_to_sat_mb

class TestConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_hc.json")

    def test_json_to_rgp(self):
        pass

    def test_extract_clauses_mb_encoder(self):
        pass

    def test_extract_clauses_er_encoder(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass