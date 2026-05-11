"""Tests for morie.fn.spflt — eigenvector spatial filtering."""
import numpy as np
from morie.fn.spflt import spatial_filter


class TestSpatialFilter:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 20
        W = np.eye(n, k=1) + np.eye(n, k=-1)
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)
        X = rng.standard_normal((n, 2))
        y = X @ [1, 0.5] + rng.standard_normal(n)
        res = spatial_filter(y, X, W, n_eigenvectors=3)
        assert res.value >= 0

    def test_r_squared_bounded(self):
        rng = np.random.default_rng(42)
        n = 15
        W = np.eye(n, k=1) + np.eye(n, k=-1)
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)
        X = rng.standard_normal((n, 1))
        y = rng.standard_normal(n)
        res = spatial_filter(y, X, W, n_eigenvectors=2)
        assert 0 <= res.extra["r_squared"] <= 1
