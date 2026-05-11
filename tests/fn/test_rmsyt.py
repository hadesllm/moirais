"""Tests for ramsey_reset."""
import numpy as np, pytest
from morie.fn.rmsyt import ramsey_reset

class TestRESET:
    def test_linear_ok(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 1))
        y = 2 * X[:, 0] + rng.normal(0, 0.5, 100)
        r = ramsey_reset(y, X)
        assert r.p_value > 0.05

    def test_nonlinear(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (100, 1))
        y = X[:, 0] ** 2 + rng.normal(0, 0.1, 100)
        r = ramsey_reset(y, X)
        assert r.test_name == "RESET"
