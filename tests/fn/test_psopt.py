"""
Tests for particle swarm optimization.
"""

import numpy as np
from moirais.fn.psopt import psopt


class TestPsopt:
    """PSO tests."""

    def test_psopt_quadratic(self):
        """Test on 2D quadratic function."""
        f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
        bounds = [(0, 5), (0, 5)]
        x_min = psopt(f, bounds, n_particles=20, generations=50, seed=42)
        assert np.allclose(x_min, [2, 3], atol=0.5)

    def test_psopt_rastrigin(self):
        """Test on Rastrigin function (multimodal)."""
        f = lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        x_min = psopt(f, bounds, n_particles=30, generations=100, seed=42)
        assert f(x_min) < 50  # Should find reasonable solution

    def test_psopt_1d(self):
        """Test on 1D function."""
        f = lambda x: (x[0] - 3.5)**2
        bounds = [(0, 10)]
        x_min = psopt(f, bounds, n_particles=15, generations=30, seed=42)
        assert np.isclose(x_min[0], 3.5, atol=1.0)

    def test_psopt_full_output(self):
        """Test full_output flag."""
        f = lambda x: np.sum(x**2)
        bounds = [(-1, 1), (-1, 1)]
        x_min, info = psopt(f, bounds, n_particles=10, generations=20, 
                           full_output=True, seed=42)
        assert 'generations' in info
        assert 'final_value' in info

    def test_psopt_custom_parameters(self):
        """Test with custom PSO parameters."""
        f = lambda x: (x[0] - 2)**2
        bounds = [(0, 5)]
        x_min = psopt(f, bounds, n_particles=15, generations=50, w=0.8, 
                     c1=2.0, c2=2.0, seed=42)
        assert x_min is not None
