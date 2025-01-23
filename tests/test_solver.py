import json
import unittest
from unittest.mock import patch, MagicMock

from src.solver import solve
from src.utils import get_file_path


class TestSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file for testing
        cls.json_file_path = get_file_path("data", "testfiles", "rgp_hc.json")

        with open(cls.json_file_path, 'r') as f:
            cls.rgp_instance = json.load(f)

    @patch('src.solver.rgp_to_sat_mb')
    @patch('src.solver.rgp_to_sat_er')
    @patch('src.solver.Solver')
    def test_solve_with_encoding_1_glucose3(self, mock_solver, mock_rgp_to_sat_er, mock_rgp_to_sat_mb):
        # Mock rgp_to_sat_mb to return a fake SAT object
        mock_rgp_to_sat_mb.return_value = {
            "final_clauses": [[1, -2, 3], [-1, 2]],
            "instance_data": {"test_key": "test_value"}
        }

        # Mock the Solver behavior
        mock_solver_instance = MagicMock()
        mock_solver_instance.solve.return_value = True
        mock_solver_instance.get_model.return_value = [1, -2, 3]
        mock_solver_instance.time.return_value = 0.5
        mock_solver_instance.accum_stats.return_value = {"conflicts": 10}
        mock_solver.return_value = mock_solver_instance

        # Call solve with Encoding 1 and Glucose3
        result = solve(enc_type=1, solver_flag=1, rgp_instance=self.rgp_instance)

        # Assertions
        self.assertTrue(result["status"])
        self.assertAlmostEqual(result["tts"], 0.5)
        self.assertEqual(result["result"], [1, -2, 3])
        self.assertFalse(result["timed_out"])
        self.assertIn("instance_data", result)

    @patch('src.solver.rgp_to_sat_mb')
    @patch('src.solver.rgp_to_sat_er')
    @patch('src.solver.Solver')
    def test_solve_with_encoding_2_lingeling(self, mock_solver, mock_rgp_to_sat_er, mock_rgp_to_sat_mb):
        # Mock rgp_to_sat_er to return a fake SAT object
        mock_rgp_to_sat_er.return_value = {
            "final_clauses": [[1, -2], [2, 3], [-1, -3]],
            "instance_data": {"test_key": "test_value"}
        }

        # Mock the Solver behavior
        mock_solver_instance = MagicMock()
        mock_solver_instance.solve.return_value = False
        mock_solver_instance.get_proof.return_value = "UNSAT Proof"
        mock_solver_instance.time.return_value = 1.2
        mock_solver_instance.accum_stats.return_value = {"conflicts": 20}
        mock_solver.return_value = mock_solver_instance

        # Call solve with Encoding 2 and Lingeling
        result = solve(enc_type=2, solver_flag=2, rgp_instance=self.rgp_instance)

        # Assertions
        self.assertFalse(result["status"])
        self.assertAlmostEqual(result["tts"], 1.2)
        self.assertEqual(result["result"], "UNSAT Proof")
        self.assertFalse(result["timed_out"])
        self.assertIn("instance_data", result)

    def test_invalid_encoding_type(self):
        with self.assertRaises(SystemExit) as cm:
            solve(enc_type=3, solver_flag=1, rgp_instance=self.rgp_instance)
        self.assertEqual(cm.exception.code, 1)

    def test_invalid_solver_flag(self):
        with self.assertRaises(SystemExit) as cm:
            solve(enc_type=1, solver_flag=3, rgp_instance=self.rgp_instance)
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()