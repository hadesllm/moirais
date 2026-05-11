"""Tests for least_median_squares."""
import numpy as np, pytest
from morie.fn.lmses import least_median_squares


class TestLMS:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (40, 1))
        y = 2.0 * X.ravel() + rng.normal(0, 0.1, 40)
        r = least_median_squares(X, y)
        assert r.measure == "lms"
        assert abs(r.extra["coefficients"][0] - 2.0) < 1.0

    def test_too_few(self):
        with pytest.raises(ValueError):
            least_median_squares(np.array([[1]]), np.array([1]))
