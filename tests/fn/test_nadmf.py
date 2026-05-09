"""
Tests for Nesterov accelerated gradient.
"""

import numpy as np
from moirais.fn.nadmf import nadmf


class TestNadmf:
    """NAG tests."""

    def test_nadmf_quadratic(self):
        """Test NAG on 2D quadratic."""
        f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
        gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
        x0 = np.array([0.0, 0.0])
        x_min = nadmf(f, gf, x0, learning_rate=0.1)
        assert np.allclose(x_min, [1, 2], atol=1e-2)

    def test_nadmf_rosenbrock(self):
        """Test NAG on Rosenbrock."""
        f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        gf = lambda x: np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])
        x0 = np.array([-1.0, -1.0])
        x_min = nadmf(f, gf, x0, learning_rate=0.001, max_iter=5000)
        assert np.allclose(x_min, [1, 1], atol=0.2)

    def test_nadmf_1d(self):
        """Test NAG on 1D."""
        f = lambda x: x[0]**2 - 4*x[0]
        gf = lambda x: np.array([2*x[0] - 4])
        x0 = np.array([0.0])
        x_min = nadmf(f, gf, x0, learning_rate=0.1)
        assert np.isclose(x_min[0], 2.0, atol=1e-2)

    def test_nadmf_momentum_effect(self):
        """Test different momentum values."""
        f = lambda x: np.sum(x**2)
        gf = lambda x: 2*x
        x0 = np.array([5.0, 5.0])
        
        x_low = nadmf(f, gf, x0, momentum=0.1, max_iter=1000)
        x_high = nadmf(f, gf, x0, momentum=0.99, max_iter=1000)
        
        # Both should reach near 0, but convergence might differ
        assert np.allclose(x_low, [0, 0], atol=1.0)
        assert np.allclose(x_high, [0, 0], atol=1.0)

    def test_nadmf_full_output(self):
        """Test full_output flag."""
        f = lambda x: x[0]**2 + x[1]**2
        gf = lambda x: 2*x
        x0 = np.array([1.0, 1.0])
        x_min, info = nadmf(f, gf, x0, full_output=True)
        assert 'iterations' in info
        assert 'converged' in info
        assert 'final_value' in info
