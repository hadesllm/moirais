"""Tests for dffits."""
import numpy as np, pytest
from morie.fn.dffts import dffits

class TestDFFITS:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (40, 2))
        y = X[:, 0] + rng.normal(0, 0.1, 40)
        r = dffits(X, y)
        assert r.name == "dffits"
        assert len(r.extra["dffits"]) == 40

    def test_outlier(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (30, 1))
        y = X[:, 0] + rng.normal(0, 0.1, 30)
        y[0] = 50
        r = dffits(X, y)
        assert r.value > 0
