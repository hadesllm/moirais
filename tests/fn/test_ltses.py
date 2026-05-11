"""Tests for least_trimmed_squares."""
import numpy as np, pytest
from morie.fn.ltses import least_trimmed_squares


class TestLTS:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (40, 1))
        y = 2.0 * X.ravel() + rng.normal(0, 0.1, 40)
        r = least_trimmed_squares(X, y)
        assert r.measure == "lts"
        assert abs(r.extra["coefficients"][0] - 2.0) < 1.0

    def test_outlier_robust(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (30, 1))
        y = X.ravel() + rng.normal(0, 0.1, 30)
        y[:3] = 100
        r = least_trimmed_squares(X, y)
        assert abs(r.extra["coefficients"][0] - 1.0) < 2.0

    def test_too_few(self):
        with pytest.raises(ValueError):
            least_trimmed_squares(np.array([[1]]), np.array([1]))
