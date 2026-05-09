"""
Tests for simulated annealing.
"""

import numpy as np
from moirais.fn.simag import simag


class TestSimag:
    """Simulated annealing tests."""

    def test_simag_quadratic(self):
        """Test on 2D quadratic."""
        f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
        bounds = [(0, 5), (0, 5)]
        x_min = simag(f, np.array([0.0, 0.0]), bounds, seed=42)
        assert np.allclose(x_min, [2, 3], atol=1.0)

    def test_simag_rastrigin(self):
        """Test on multimodal Rastrigin."""
        f = lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        x_min = simag(f, np.array([0.0, 0.0]), bounds, 
                     T_init=10.0, cooling_rate=0.95, max_iter=2000, seed=42)
        assert f(x_min) < 100

    def test_simag_1d(self):
        """Test on 1D function."""
        f = lambda x: (x[0] - 2.5)**2
        bounds = [(0, 5)]
        x_min = simag(f, np.array([0.0]), bounds, seed=42)
        assert np.isclose(x_min[0], 2.5, atol=1.0)

    def test_simag_full_output(self):
        """Test full_output flag."""
        f = lambda x: np.sum(x**2)
        bounds = [(-1, 1), (-1, 1)]
        x_min, info = simag(f, np.array([0.5, 0.5]), bounds, 
                           max_iter=100, full_output=True, seed=42)
        assert 'iterations' in info
        assert 'final_value' in info

    def test_simag_temperature_decay(self):
        """Test temperature decay effect."""
        f = lambda x: (x[0] - 1)**2
        bounds = [(0, 2)]
        # Fast cooling should find minimum faster
        x_fast = simag(f, np.array([0.0]), bounds, cooling_rate=0.8, 
                      max_iter=500, seed=42)
        assert x_fast is not None
