import os
import unittest

from src.utils import get_file_path


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(__file__)

    def test_get_file_path_valid_data(self):
        # Test for valid 'data' target directory
        target_dir = "data"
        foldername = "config"
        filename = "testfile.txt"

        expected_path = os.path.abspath(os.path.join(self.base_dir, '..', target_dir, foldername, filename))
        self.assertEqual(get_file_path(target_dir, foldername, filename), expected_path)

    def test_get_file_path_valid_assets(self):
        # Test for valid 'assets' target directory
        src_dir = "src"
        target_dir = "assets"
        foldername = "images"
        filename = "image.png"

        expected_path = os.path.abspath(os.path.join(self.base_dir, '..', src_dir, target_dir, foldername, filename))
        self.assertEqual(get_file_path(target_dir, foldername, filename), expected_path)

    def test_get_file_path_invalid_target_dir(self):
        # Test for invalid target directory
        target_dir = "invalid"
        foldername = "config"
        filename = "testfile.txt"

        with self.assertRaises(SystemExit) as cm:
            get_file_path(target_dir, foldername, filename)
        self.assertEqual(cm.exception.code, 1)

    def test_get_file_path_non_string_target_dir(self):
        # Test for non-string target directory
        target_dir = 123
        foldername = "config"
        filename = "testfile.txt"

        with self.assertRaises(SystemExit) as cm:
            get_file_path(target_dir, foldername, filename)
        self.assertEqual(cm.exception.code, 1)

    def test_get_file_path_empty_foldername(self):
        # Test with an empty folder name
        target_dir = "data"
        foldername = ""
        filename = "testfile.txt"

        expected_path = os.path.abspath(os.path.join(self.base_dir, '..', target_dir, filename))
        self.assertEqual(get_file_path(target_dir, foldername, filename), expected_path)

    def test_get_file_path_empty_filename(self):
        # Test with an empty filename
        target_dir = "data"
        foldername = "config"
        filename = ""

        expected_path = os.path.abspath(os.path.join(self.base_dir, '..', target_dir, foldername))
        self.assertEqual(os.path.normpath(get_file_path(target_dir, foldername, filename)), expected_path)


if __name__ == "__main__":
    unittest.main()