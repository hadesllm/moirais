"""Tests for morie.fn.white -- White's test."""

import numpy as np
import pytest
from morie.fn.white import white_test
from morie.fn._containers import TestResult


class TestWhite:
    def test_homoscedastic(self):
        """Constant variance => non-significant."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 1))
        y = 3 * X[:, 0] + rng.normal(0, 1, n)
        Xa = np.column_stack([np.ones(n), X])
        beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        resid = y - Xa @ beta
        r = white_test(resid, X)
        assert isinstance(r, TestResult)
        assert r.p_value > 0.05

    def test_heteroscedastic(self):
        """Variance depends on X^2 => significant."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.uniform(1, 10, (n, 1))
        y = 2 * X[:, 0] + rng.normal(0, 1, n) * X[:, 0]
        Xa = np.column_stack([np.ones(n), X])
        beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        resid = y - Xa @ beta
        r = white_test(resid, X)
        assert r.p_value < 0.05

    def test_raises_dim_mismatch(self):
        with pytest.raises(ValueError):
            white_test([1, 2], [[1], [2], [3]])
