"""Tests for morie.fn.newyw — Newey-West HAC SE."""

import numpy as np
import pytest

from morie.fn.newyw import newey_west


def test_nw_coefficients_same_as_ols():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = 2.0 + 3.0 * X[:, 0] + rng.standard_normal(n) * 0.5
    from morie.fn.olsrg import ols_regression
    ols = ols_regression(y, X)
    nw = newey_west(y, X)
    np.testing.assert_allclose(
        list(ols.coefficients.values()),
        list(nw.coefficients.values()),
        atol=1e-10,
    )


def test_nw_se_differs_from_ols_with_autocorrelation():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    e = np.zeros(n)
    e[0] = rng.standard_normal()
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.standard_normal()
    y = 1.0 + 2.0 * X[:, 0] + e
    from morie.fn.olsrg import ols_regression
    ols = ols_regression(y, X)
    nw = newey_west(y, X, max_lag=5)
    ols_se = list(ols.se.values())
    nw_se = list(nw.se.values())
    assert not np.allclose(ols_se, nw_se, atol=1e-4)


def test_nw_auto_lag():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 1))
    y = rng.standard_normal(100)
    res = newey_west(y, X)
    assert res.extra["max_lag"] >= 0
