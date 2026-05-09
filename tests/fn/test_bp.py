"""Tests for moirais.fn.bp -- Breusch-Pagan test."""

import numpy as np
import pytest
from moirais.fn.bp import breusch_pagan_test
from moirais.fn._containers import TestResult


class TestBreuschPagan:
    def test_homoscedastic(self):
        """Constant variance => non-significant."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 1))
        y = 2 * X[:, 0] + rng.normal(0, 1, n)
        beta = np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)[0]
        resid = y - np.column_stack([np.ones(n), X]) @ beta
        r = breusch_pagan_test(resid, X)
        assert isinstance(r, TestResult)
        assert r.p_value > 0.05

    def test_heteroscedastic(self):
        """Variance proportional to X => significant."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.uniform(1, 10, (n, 1))
        y = 2 * X[:, 0] + rng.normal(0, 1, n) * X[:, 0]
        beta = np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)[0]
        resid = y - np.column_stack([np.ones(n), X]) @ beta
        r = breusch_pagan_test(resid, X)
        assert r.p_value < 0.05

    def test_raises_dim_mismatch(self):
        with pytest.raises(ValueError):
            breusch_pagan_test([1, 2, 3], [[1, 2], [3, 4]])
