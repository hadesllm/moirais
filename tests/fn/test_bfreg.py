"""Tests for biweight_regression."""
import numpy as np, pytest
from moirais.fn.bfreg import biweight_regression


class TestBiweightRegression:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 1))
        y = 3.0 * X.ravel() + rng.normal(0, 0.1, 50)
        r = biweight_regression(X, y)
        assert r.measure == "biweight_regression"
        assert r.extra["coefficients"][0] == pytest.approx(3.0, abs=0.5)

    def test_outlier_resistance(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (40, 1))
        y = 2.0 * X.ravel() + rng.normal(0, 0.1, 40)
        y[0] = 200
        r = biweight_regression(X, y)
        assert abs(r.extra["coefficients"][0] - 2.0) < 2.0

    def test_too_few(self):
        with pytest.raises(ValueError):
            biweight_regression(np.array([[1]]), np.array([1]))
