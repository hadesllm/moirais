"""Tests for least_median_sq."""
import numpy as np, pytest
from moirais.fn.lms import least_median_sq

class TestLMS:
    def test_basic(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (40, 1))
        y = X[:, 0] * 2 + 1 + rng.normal(0, 0.1, 40)
        r = least_median_sq(X, y, seed=2)
        assert r.name == "lms"
        assert r.value >= 0

    def test_returns_coefficients(self):
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        r = least_median_sq(X, y, seed=0)
        assert len(r.extra["coefficients"]) == 2
