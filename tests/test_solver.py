import unittest

from src.solver import solve
from src.utils import get_file_path

class TestSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_hc.json")

    def test_solve(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass