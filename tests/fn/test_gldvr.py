"""Tests for goldfeld_quandt_test."""
import numpy as np, pytest
from morie.fn.gldvr import goldfeld_quandt_test


class TestGoldfeldQuandt:
    def test_homoscedastic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 1))
        y = 2.0 * X.ravel() + rng.normal(0, 1, 60)
        r = goldfeld_quandt_test(X, y)
        assert r.test_name == "Goldfeld-Quandt test"
        assert r.p_value > 0.05

    def test_heteroscedastic(self):
        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(1, 10, (60, 1)), axis=0)
        y = 2.0 * X.ravel() + rng.normal(0, 1, 60) * X.ravel()
        r = goldfeld_quandt_test(X, y)
        assert r.p_value < 0.2
