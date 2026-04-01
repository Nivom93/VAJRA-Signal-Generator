import sys
from unittest.mock import MagicMock, patch

# --- Dependency Mocking ---
# We use a context-safe approach for mocking to avoid global side effects
# during imports if possible, but since vajra_engine_ultra_v6_final.py
# executes code on import (e.g., logging setup), we must ensure
# dependencies are available.

mock_np = MagicMock()
mock_np.float64 = float
mock_np.uint8 = int
mock_np.int64 = int
mock_np.nan = float('nan')
mock_np.inf = float('inf')

mock_pd = MagicMock()

# Safely mock numba.njit
def mock_njit(*args, **kwargs):
    def decorator(func):
        return func
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

mock_numba = MagicMock()
mock_numba.njit = mock_njit

# Apply mocks to sys.modules for the duration of the test module
MOCK_MODULES = {
    'numpy': mock_np,
    'pandas': mock_pd,
    'ccxt': MagicMock(),
    'joblib': MagicMock(),
    'numba': mock_numba
}

with patch.dict(sys.modules, MOCK_MODULES):
    import unittest
    from vajra_engine_ultra_v6_final import _safe_divide

class TestSafeDivide(unittest.TestCase):
    def test_safe_divide_normal(self):
        """Test division with standard positive and negative numbers."""
        self.assertEqual(_safe_divide(10, 2), 5.0)
        self.assertEqual(_safe_divide(-10, 2), -5.0)
        self.assertEqual(_safe_divide(10, -2), -5.0)
        self.assertEqual(_safe_divide(0, 5), 0.0)

    def test_safe_divide_by_zero(self):
        """Test that division by exactly zero returns the default value."""
        self.assertEqual(_safe_divide(10, 0), 0.0)
        self.assertEqual(_safe_divide(10, 0, default=99.0), 99.0)

    def test_safe_divide_near_zero(self):
        """
        Test that division by values smaller than the epsilon (1e-12)
        returns the default value.
        """
        # Threshold in vajra_engine_ultra_v6_final.py is 1e-12
        self.assertEqual(_safe_divide(10, 1e-13), 0.0)
        self.assertEqual(_safe_divide(10, -1e-13), 0.0)
        self.assertEqual(_safe_divide(10, 1e-13, default=-1.0), -1.0)

    def test_safe_divide_at_threshold(self):
        """
        Test boundary conditions around the 1e-12 epsilon.
        """
        # Exactly 1e-12 should NOT be treated as zero if the check is abs(b) < 1e-12
        self.assertAlmostEqual(_safe_divide(1, 1e-12), 1e12)
        # Slightly above threshold
        self.assertAlmostEqual(_safe_divide(1, 1.1e-12), 1/1.1e-12)

if __name__ == '__main__':
    unittest.main()
