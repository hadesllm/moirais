"""Tests for morie.fn.punsh -- elastic net regularization."""

import numpy as np
from morie.fn.punsh import penalty_regression, punsh
from morie.fn._containers import RegressionResult


class TestPunsh:
    def test_alias(self):
        assert punsh is penalty_regression

    def test_lasso(self):
        rng = np.random.default_rng(42)
        n, p = 100, 5
        X = rng.normal(0, 1, (n, p))
        beta = np.array([3, 0, 0, -2, 0], dtype=float)
        y = X @ beta + rng.normal(0, 0.5, n)
        r = penalty_regression(X, y, alpha=0.5, l1_ratio=1.0)
        assert isinstance(r, RegressionResult)
        assert r.extra["n_nonzero"] <= 5

    def test_ridge(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 3))
        y = X @ [1, 2, 3] + rng.normal(0, 0.1, 50)
        r = penalty_regression(X, y, alpha=0.01, l1_ratio=0.0)
        assert r.r_squared > 0.9
