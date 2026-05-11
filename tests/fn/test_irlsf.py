"""
Tests for IRLS GLM fitting.
"""

import numpy as np
from morie.fn.irlsf import irlsf


class TestIrlsf:
    """IRLS tests."""

    def test_irlsf_gaussian(self):
        """Test IRLS on Gaussian (linear regression)."""
        np.random.seed(42)
        X = np.column_stack([np.ones(20), np.linspace(0, 1, 20)])
        y = 1 + 2*X[:, 1] + np.random.randn(20)*0.1
        beta = irlsf(X, y, family='gaussian')
        assert np.allclose(beta, [1, 2], atol=0.3)

    def test_irlsf_poisson(self):
        """Test IRLS on Poisson GLM."""
        np.random.seed(42)
        X = np.column_stack([np.ones(30), np.random.randn(30)])
        y_mean = np.exp(0.5 + 0.3*X[:, 1])
        y = np.random.poisson(y_mean)
        beta = irlsf(X, y, family='poisson')
        assert len(beta) == 2

    def test_irlsf_binomial(self):
        """Test IRLS on binomial GLM (logistic regression)."""
        np.random.seed(42)
        X = np.column_stack([np.ones(50), np.random.randn(50)])
        prob = 1.0 / (1.0 + np.exp(-(-0.5 + 0.5*X[:, 1])))
        y = (np.random.rand(50) < prob).astype(int)
        beta = irlsf(X, y, family='binomial')
        assert len(beta) == 2

    def test_irlsf_full_output(self):
        """Test full_output flag."""
        X = np.column_stack([np.ones(10), np.linspace(0, 1, 10)])
        y = 1 + 2*X[:, 1]
        beta, info = irlsf(X, y, family='gaussian', full_output=True)
        assert 'iterations' in info
        assert 'converged' in info
        assert 'deviance' in info

    def test_irlsf_intercept_only(self):
        """Test intercept-only model."""
        X = np.ones((20, 1))
        y = np.ones(20) * 3
        beta = irlsf(X, y, family='gaussian')
        assert np.isclose(beta[0], 3.0, atol=0.5)
