"""Tests for morie.fn.lasrg — LASSO regression."""

import numpy as np
import pytest

from morie.fn.lasrg import lasso_regression


def test_lasso_sparsity():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 10))
    y = 3.0 * X[:, 0] + rng.standard_normal(n) * 0.5
    res = lasso_regression(y, X, lam=0.5)
    n_zero = sum(1 for k, v in res.coefficients.items()
                 if k != "(Intercept)" and abs(v) < 1e-8)
    assert n_zero >= 5


def test_lasso_recovers_signal():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 5))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.standard_normal(n) * 0.3
    res = lasso_regression(y, X, lam=0.05)
    assert abs(res.coefficients["x0"]) > 1.0
    assert abs(res.coefficients["x1"]) > 0.5


def test_lasso_high_penalty_zeros_all():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    res = lasso_regression(y, X, lam=100.0)
    for k, v in res.coefficients.items():
        if k != "(Intercept)":
            assert abs(v) < 0.01


def test_lasso_r_squared():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    y = 5.0 * X[:, 0] + rng.standard_normal(100) * 0.1
    res = lasso_regression(y, X, lam=0.01)
    assert res.r_squared > 0.9
