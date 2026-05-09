"""
Tests for Nelder-Mead simplex optimization.
"""

import numpy as np
from moirais.fn.nmead import nmead


class TestNmead:
    """Nelder-Mead tests."""

    def test_nmead_quadratic_2d(self):
        """Test on 2D quadratic function."""
        f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
        x0 = np.array([0.0, 0.0])
        x_min = nmead(f, x0)
        assert np.allclose(x_min, [2, 3], atol=1e-4)

    def test_nmead_rosenbrock(self):
        """Test on Rosenbrock function."""
        f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        x0 = np.array([-1.0, -1.0])
        x_min = nmead(f, x0, max_iter=2000)
        assert np.allclose(x_min, [1, 1], atol=0.1)

    def test_nmead_1d(self):
        """Test on 1D function."""
        f = lambda x: (x[0] - 5)**2
        x0 = np.array([0.0])
        x_min = nmead(f, x0)
        assert np.isclose(x_min[0], 5.0, atol=1e-3)

    def test_nmead_full_output(self):
        """Test full_output flag."""
        f = lambda x: x[0]**2 + x[1]**2
        x0 = np.array([1.0, 1.0])
        x_min, info = nmead(f, x0, full_output=True)
        assert 'iterations' in info
        assert 'converged' in info
        assert 'final_value' in info

    def test_nmead_3d(self):
        """Test on 3D function."""
        f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2
        x0 = np.array([0.0, 0.0, 0.0])
        x_min = nmead(f, x0)
        assert np.allclose(x_min, [1, 2, 3], atol=1e-3)
