"""Tests for ramsey_reset_test."""
import numpy as np, pytest
from morie.fn.ramsy import ramsey_reset_test


class TestRamseyRESET:
    def test_linear_ok(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 1))
        y = 3.0 * X.ravel() + rng.normal(0, 0.5, 60)
        r = ramsey_reset_test(X, y)
        assert r.test_name == "Ramsey RESET test"
        assert r.p_value > 0.05

    def test_nonlinear(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 5, (60, 1))
        y = np.sin(X.ravel()) + rng.normal(0, 0.1, 60)
        r = ramsey_reset_test(X, y)
        assert r.p_value < 0.5
