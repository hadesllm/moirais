"""
Tests for L-BFGS quasi-Newton method.
"""

import numpy as np
from morie.fn.lbfgs import lbfgs


class TestLbfgs:
    """L-BFGS tests."""

    def test_lbfgs_quadratic_2d(self):
        """Test on 2D quadratic."""
        f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
        gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
        x0 = np.array([0.0, 0.0])
        x_min = lbfgs(f, gf, x0)
        assert np.allclose(x_min, [1, 2], atol=1e-4)

    def test_lbfgs_rosenbrock(self):
        """Test on Rosenbrock function."""
        f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        gf = lambda x: np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])
        x0 = np.array([-1.0, -1.0])
        x_min = lbfgs(f, gf, x0, max_iter=500)
        assert np.allclose(x_min, [1, 1], atol=0.1)

    def test_lbfgs_memory_parameter(self):
        """Test with different memory parameter m."""
        f = lambda x: np.sum(x**2)
        gf = lambda x: 2*x
        x0 = np.array([1.0, 2.0, 3.0])
        x_min = lbfgs(f, gf, x0, m=5)
        assert np.allclose(x_min, [0, 0, 0], atol=1e-4)

    def test_lbfgs_full_output(self):
        """Test full_output flag."""
        f = lambda x: x[0]**2 + x[1]**2
        gf = lambda x: 2*x
        x0 = np.array([1.0, 1.0])
        x_min, info = lbfgs(f, gf, x0, full_output=True)
        assert 'iterations' in info
        assert 'converged' in info
