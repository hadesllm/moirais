"""Tests for moirais.fn.spdur — spatial Durbin model."""
import numpy as np
from moirais.fn.spdur import spatial_durbin


class TestSpatialDurbin:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 20
        W = np.eye(n, k=1) + np.eye(n, k=-1)
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)
        X = rng.standard_normal((n, 2))
        y = X @ [1, 0.5] + rng.standard_normal(n)
        res = spatial_durbin(y, X, W)
        assert "beta" in res.extra
        assert "theta" in res.extra

    def test_rho_bounded(self):
        rng = np.random.default_rng(42)
        n = 15
        W = np.eye(n, k=1) + np.eye(n, k=-1)
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)
        X = rng.standard_normal((n, 1))
        y = rng.standard_normal(n)
        res = spatial_durbin(y, X, W)
        assert -1 < res.extra["rho"] < 1
