import unittest

from src.helper import get_file_path
from src.er_encoder import rgp_to_sat_er

class TestEREncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_test_small.json")


    @classmethod
    def tearDownClass(cls):
        pass