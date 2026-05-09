"""Tests for moirais.fn.rigdg — Ridge regression."""

import numpy as np
import pytest

from moirais.fn.rigdg import ridge_regression


def test_ridge_shrinks_coefficients():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.standard_normal(n) * 0.1
    res_low = ridge_regression(y, X, lam=0.01)
    res_high = ridge_regression(y, X, lam=100.0)
    ols_norm = sum(v ** 2 for k, v in res_low.coefficients.items() if k != "(Intercept)")
    ridge_norm = sum(v ** 2 for k, v in res_high.coefficients.items() if k != "(Intercept)")
    assert ridge_norm < ols_norm


def test_ridge_lam_zero_near_ols():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = 2.0 * X[:, 0] + rng.standard_normal(n) * 0.1
    from moirais.fn.olsrg import ols_regression
    ols = ols_regression(y, X)
    ridge = ridge_regression(y, X, lam=1e-10)
    assert abs(ols.coefficients["x0"] - ridge.coefficients["x0"]) < 0.01


def test_ridge_negative_lam_raises():
    with pytest.raises(ValueError, match="non-negative"):
        ridge_regression(np.ones(5), np.ones((5, 1)), lam=-1)


def test_ridge_gcv():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = X @ np.array([1, 2, 3]) + rng.standard_normal(50)
    res = ridge_regression(y, X, lam=1.0)
    assert np.isfinite(res.extra["gcv"])
    assert res.extra["eff_df"] > 0
