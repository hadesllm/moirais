"""Tests for mm_estimator."""
import numpy as np, pytest
from morie.fn.mmslt import mm_estimator


class TestMMEstimator:
    def test_basic_regression(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 2))
        y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.1, 50)
        r = mm_estimator(X, y)
        assert r.measure == "mm_estimator"
        assert r.extra["coefficients"][0] == pytest.approx(2.0, abs=0.5)

    def test_outlier_resistance(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (30, 1))
        y = 3.0 * X.ravel() + rng.normal(0, 0.1, 30)
        y[0] = 100
        r = mm_estimator(X, y)
        assert abs(r.extra["coefficients"][0] - 3.0) < 2.0

    def test_too_few(self):
        with pytest.raises(ValueError):
            mm_estimator(np.array([[1]]), np.array([1]))
