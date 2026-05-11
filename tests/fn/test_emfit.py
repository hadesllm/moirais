"""
Tests for EM algorithm.
"""

import numpy as np
from morie.fn.emfit import emfit


class TestEmfit:
    """EM algorithm tests."""

    def test_emfit_gaussian_mixture_2d(self):
        """Test fitting 2-component Gaussian mixture."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2) + np.array([0, 0])
        X2 = np.random.randn(50, 2) + np.array([5, 5])
        X = np.vstack([X1, X2])

        params, info = emfit(X, n_components=2, max_iter=100, seed=42, full_output=True)
        assert 'means' in params
        assert 'covars' in params
        assert 'weights' in params
        assert params['means'].shape == (2, 2)

    def test_emfit_single_component(self):
        """Test fitting single component."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        params, _ = emfit(X, n_components=1, max_iter=50, seed=42, full_output=True)
        assert np.isclose(params['means'][0, 0], 0, atol=0.5)

    def test_emfit_convergence(self):
        """Test convergence flag."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        params1, info1 = emfit(X, n_components=2, max_iter=200, tol=1e-6, 
                              seed=42, full_output=True)
        params2, info2 = emfit(X, n_components=2, max_iter=200, tol=1e-3, 
                              seed=42, full_output=True)
        # Should converge faster with higher tolerance
        assert info2['iterations'] <= info1['iterations']

    def test_emfit_3d(self):
        """Test on 3D data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        params, info = emfit(X, n_components=2, max_iter=100, seed=42, full_output=True)
        assert params['means'].shape == (2, 3)
        assert params['covars'].shape == (2, 3, 3)

    def test_emfit_weights_sum_to_one(self):
        """Test that mixing weights sum to 1."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        params, _ = emfit(X, n_components=3, seed=42, full_output=True)
        assert np.isclose(np.sum(params['weights']), 1.0)
