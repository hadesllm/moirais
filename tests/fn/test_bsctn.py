"""
Tests for bisection method.
"""

import numpy as np
import pytest
from morie.fn.bsctn import bsctn


class TestBsctn:
    """Bisection method tests."""

    def test_bsctn_quadratic(self):
        """Test bisection on x^2 - 2."""
        f = lambda x: x**2 - 2
        root = bsctn(f, 1.0, 2.0)
        assert np.isclose(root, np.sqrt(2), atol=1e-6)

    def test_bsctn_cubic(self):
        """Test bisection on x^3 - 8."""
        f = lambda x: x**3 - 8
        root = bsctn(f, 1.0, 3.0)
        assert np.isclose(root, 2.0, atol=1e-5)

    def test_bsctn_sine(self):
        """Test bisection on sin(x)."""
        f = lambda x: np.sin(x)
        root = bsctn(f, 3.0, 4.0)
        assert np.isclose(root, np.pi, atol=1e-5)

    def test_bsctn_full_output(self):
        """Test full_output flag."""
        f = lambda x: x - 2
        root, info = bsctn(f, 0.0, 3.0, full_output=True)
        assert 'iterations' in info
        assert 'converged' in info
        assert np.isclose(root, 2.0, atol=1e-5)

    def test_bsctn_sign_error(self):
        """Test error when f(a) and f(b) have same sign."""
        f = lambda x: x**2
        with pytest.raises(ValueError):
            bsctn(f, 1.0, 2.0)

    def test_bsctn_negative_root(self):
        """Test finding negative root."""
        f = lambda x: x + 3
        root = bsctn(f, -5.0, 0.0)
        assert np.isclose(root, -3.0, atol=1e-5)
