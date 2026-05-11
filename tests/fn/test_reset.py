"""Tests for morie.fn.reset -- Ramsey RESET test."""

import numpy as np
import pytest
from morie.fn.reset import ramsey_reset_test
from morie.fn._containers import TestResult


class TestReset:
    def test_linear_model_correct(self):
        """Truly linear DGP => RESET should not reject."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = 2 + 3 * x + rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        r = ramsey_reset_test(y, X, fitted)
        assert isinstance(r, TestResult)
        assert r.p_value > 0.05

    def test_nonlinear_misspecification(self):
        """Quadratic DGP with linear model => RESET rejects."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        y = 1 + 2 * x + 3 * x ** 2 + rng.normal(0, 0.5, n)
        X = np.column_stack([np.ones(n), x])  # omits x^2
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        r = ramsey_reset_test(y, X, fitted)
        assert r.p_value < 0.05

    def test_raises_dim_mismatch(self):
        with pytest.raises(ValueError):
            ramsey_reset_test([1, 2], [[1], [2], [3]], [1, 2])
