import unittest

from src.helper import get_file_path
from src.generator import (
    generate_partition,
    partition_array_into_k_groups,
    find_num_groups_involved,
    generate_constraints,
    generate_rgp_instances
)

class TestFormatValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_file_path = get_file_path("testfiles", "rgp_hc.json")

    def test_generate_partition(self):
        pass

    def test_partition_array_into_k_groups(self):
        pass

    def test_find_num_groups_involved(self):
        pass

    def test_generate_constraints(self):
        pass

    def test_generate_rgp_instances(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass