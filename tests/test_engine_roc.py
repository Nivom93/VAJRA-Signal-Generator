import unittest
import numpy as np
from vajra_engine_ultra_v6_final import _roc

class TestEngineROC(unittest.TestCase):
    def test_roc_normal(self):
        # Standard case: valid array and k
        arr = np.array([10.0, 20.0, 40.0], dtype=np.float64)
        k = 1
        expected = np.array([0.0, 1.0, 1.0], dtype=np.float64)  # (20/10)-1=1, (40/20)-1=1
        result = _roc(arr, k)
        np.testing.assert_array_almost_equal(result, expected)

    def test_roc_k_zero(self):
        # Edge case: k = 0
        arr = np.array([10.0, 20.0, 40.0], dtype=np.float64)
        k = 0
        expected = np.zeros_like(arr, dtype=np.float64)
        result = _roc(arr, k)
        np.testing.assert_array_almost_equal(result, expected)

    def test_roc_k_negative(self):
        # Edge case: k < 0
        arr = np.array([10.0, 20.0, 40.0], dtype=np.float64)
        k = -1
        expected = np.zeros_like(arr, dtype=np.float64)
        result = _roc(arr, k)
        np.testing.assert_array_almost_equal(result, expected)

    def test_roc_k_equal_size(self):
        # Edge case: arr.size == k
        arr = np.array([10.0, 20.0, 40.0], dtype=np.float64)
        k = 3
        expected = np.zeros_like(arr, dtype=np.float64)
        result = _roc(arr, k)
        np.testing.assert_array_almost_equal(result, expected)

    def test_roc_k_greater_than_size(self):
        # Edge case: arr.size < k
        arr = np.array([10.0, 20.0, 40.0], dtype=np.float64)
        k = 5
        expected = np.zeros_like(arr, dtype=np.float64)
        result = _roc(arr, k)
        np.testing.assert_array_almost_equal(result, expected)

    def test_roc_division_by_zero(self):
        # Edge case: shifted contains 0
        arr = np.array([0.0, 10.0, 20.0], dtype=np.float64)
        k = 1
        # shifted = [nan, 0, 10]
        # arr / shifted = [nan, inf, 2]
        # (arr / shifted) - 1.0 = [nan, inf, 1]
        # np.nan_to_num = [0, np.finfo(np.float64).max, 1]
        result = _roc(arr, k)
        self.assertEqual(result[0], 0.0)
        # Verify result[1] is the maximum possible float64 value (infinity replacement)
        self.assertEqual(result[1], np.finfo(np.float64).max)
        self.assertEqual(result[2], 1.0)

    def test_roc_zero_divided_by_zero(self):
        # Edge case: zero divided by zero
        arr = np.array([0.0, 0.0, 20.0], dtype=np.float64)
        k = 1
        # shifted = [nan, 0, 0]
        # arr / shifted = [nan, nan, inf]
        # result = [0, 0, np.finfo(np.float64).max]
        result = _roc(arr, k)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[2], np.finfo(np.float64).max)

    def test_roc_all_zeros(self):
        # Edge case: all zeros in input
        arr = np.zeros(5, dtype=np.float64)
        k = 2
        result = _roc(arr, k)
        expected = np.zeros(5, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
