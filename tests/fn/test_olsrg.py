"""Tests for morie.fn.olsrg — OLS regression."""

import numpy as np
import pytest

from morie.fn.olsrg import ols_regression


@pytest.fixture()
def ols_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    y = 3.0 + 1.5 * X[:, 0] - 2.0 * X[:, 1] + rng.standard_normal(n) * 0.5
    return y, X


def test_coefficients_near_true(ols_data):
    y, X = ols_data
    res = ols_regression(y, X)
    assert abs(res.coefficients["(Intercept)"] - 3.0) < 0.2
    assert abs(res.coefficients["x0"] - 1.5) < 0.2
    assert abs(res.coefficients["x1"] - (-2.0)) < 0.2


def test_r_squared_high(ols_data):
    y, X = ols_data
    res = ols_regression(y, X)
    assert res.r_squared > 0.9


def test_p_values_significant(ols_data):
    y, X = ols_data
    res = ols_regression(y, X)
    assert res.p_values["x0"] < 0.05
    assert res.p_values["x1"] < 0.05


def test_f_statistic(ols_data):
    y, X = ols_data
    res = ols_regression(y, X)
    assert res.extra["f_statistic"] > 10
    assert res.extra["f_pvalue"] < 0.001


def test_residuals_sum_near_zero(ols_data):
    y, X = ols_data
    res = ols_regression(y, X)
    assert abs(np.sum(res.residuals)) < 1e-8


def test_no_intercept():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 1))
    y = 2.0 * X[:, 0] + rng.standard_normal(50) * 0.1
    res = ols_regression(y, X, add_intercept=False)
    assert "(Intercept)" not in res.coefficients
    assert abs(res.coefficients["x0"] - 2.0) < 0.2


def test_singular_raises():
    X = np.ones((10, 2))
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="singular"):
        ols_regression(y, X)
