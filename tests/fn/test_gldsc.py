"""
Tests for golden section search.
"""

import numpy as np
from moirais.fn.gldsc import gldsc


class TestGldsc:
    """Golden section search tests."""

    def test_gldsc_quadratic(self):
        """Test on quadratic function (x-3)^2."""
        f = lambda x: (x - 3)**2
        x_min = gldsc(f, 0.0, 5.0)
        assert np.isclose(x_min, 3.0, atol=1e-5)

    def test_gldsc_cubic(self):
        """Test on cubic with minimum at x=2."""
        f = lambda x: (x - 2)**3 + (x - 2)**2
        x_min = gldsc(f, -1.0, 4.0)
        assert x_min < 4.0  # Should be near 2

    def test_gldsc_sine(self):
        """Test on sine squared."""
        f = lambda x: np.sin(x)**2
        x_min = gldsc(f, 0.0, np.pi)
        assert x_min > 0 and x_min < np.pi

    def test_gldsc_full_output(self):
        """Test full_output flag."""
        f = lambda x: (x + 2)**2
        x_min, info = gldsc(f, -5.0, 0.0, full_output=True)
        assert 'iterations' in info
        assert 'converged' in info
        assert 'final_value' in info
        assert np.isclose(x_min, -2.0, atol=1e-4)

    def test_gldsc_exponential(self):
        """Test on exponential function."""
        f = lambda x: np.exp(x) + np.exp(-x)
        x_min = gldsc(f, -2.0, 2.0)
        assert np.isclose(x_min, 0.0, atol=1e-5)
