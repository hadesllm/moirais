"""Tests for morie.fn.skerr — spatial error model (SEM)."""

import numpy as np
import pytest

from morie.fn.skerr import spatial_error_model


class TestSpatialErrorModel:

    def test_lambda_estimated(self):
        """Lambda (spatial autocorrelation) is estimated and finite."""
        rng = np.random.default_rng(42)
        n = 30
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i + 1] = 0.5
            W[i + 1, i] = 0.5
        lam_true = 0.5
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta_true = np.array([1.0, 2.0])
        eps = rng.standard_normal(n) * 0.5
        u = np.linalg.solve(np.eye(n) - lam_true * W, eps)
        y = X @ beta_true + u
        result = spatial_error_model(y, X, W)
        lam_hat = result.extra["lambda_"]
        assert np.isfinite(lam_hat)
        assert "coefs" in result.extra

    def test_coefficients_shape(self):
        """Coefficients have correct shape (p,)."""
        n, p = 20, 3
        rng = np.random.default_rng(99)
        X = rng.standard_normal((n, p))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(n) * 0.1
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i + 1] = 1.0
            W[i + 1, i] = 1.0
        result = spatial_error_model(y, X, W)
        assert result.extra["coefs"].shape == (p,)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError):
            spatial_error_model(np.array([1, 2, 3]), np.ones((4, 1)), np.eye(3))
