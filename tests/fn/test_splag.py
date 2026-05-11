"""Tests for morie.fn.splag — spatial lag model."""
import numpy as np
import pytest
from morie.fn.splag import spatial_lag


class TestSpatialLag:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 30
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(max(0, i-1), min(n, i+2)):
                if i != j:
                    W[i, j] = 1
        W = W / W.sum(axis=1, keepdims=True)
        X = rng.standard_normal((n, 2))
        y = X @ [1.0, 0.5] + 0.3 * W @ (X @ [1.0, 0.5]) + rng.standard_normal(n) * 0.5
        res = spatial_lag(y, X, W)
        assert "coefficients" in res.extra

    def test_returns_rho(self):
        rng = np.random.default_rng(42)
        n = 20
        W = np.eye(n, k=1) + np.eye(n, k=-1)
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)
        X = rng.standard_normal((n, 1))
        y = X.ravel() + rng.standard_normal(n)
        res = spatial_lag(y, X, W)
        assert "rho" in res.extra
