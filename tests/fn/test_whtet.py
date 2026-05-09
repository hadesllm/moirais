"""Tests for white_heterosc_test."""
import numpy as np, pytest
from moirais.fn.whtet import white_heterosc_test


class TestWhiteTest:
    def test_homoscedastic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (80, 2))
        y = X @ np.array([1, 2]) + rng.normal(0, 1, 80)
        r = white_heterosc_test(X, y)
        assert r.test_name == "White's test"
        assert r.p_value > 0.05

    def test_keys(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (80, 1))
        y = X.ravel() + rng.normal(0, 1, 80)
        r = white_heterosc_test(X, y)
        assert "r_squared_aux" in r.extra
