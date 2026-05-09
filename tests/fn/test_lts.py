"""Tests for least_trimmed_sq."""
import numpy as np, pytest
from moirais.fn.lts import least_trimmed_sq

class TestLTS:
    def test_linear(self):
        rng = np.random.default_rng(5)
        X = rng.normal(0, 1, (50, 1))
        y = 2 * X[:, 0] + 1 + rng.normal(0, 0.1, 50)
        r = least_trimmed_sq(X, y, seed=5)
        assert r.name == "lts"
        assert r.extra["coefficients"][1] == pytest.approx(2.0, abs=0.5)

    def test_with_outliers(self):
        rng = np.random.default_rng(6)
        X = rng.normal(0, 1, (50, 1))
        y = 3 * X[:, 0] + rng.normal(0, 0.1, 50)
        y[:5] = 100
        r = least_trimmed_sq(X, y, seed=6)
        assert abs(r.extra["coefficients"][1]) < 10
