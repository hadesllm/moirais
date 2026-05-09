"""Tests for moirais.fn.wlsrg — WLS regression."""

import numpy as np
import pytest

from moirais.fn.wlsrg import wls_regression


def test_wls_recovers_coefficients():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 1))
    sigma = 0.5 + 2.0 * np.abs(X[:, 0])
    y = 1.0 + 3.0 * X[:, 0] + rng.standard_normal(n) * sigma
    w = 1.0 / (sigma ** 2)
    res = wls_regression(y, X, w)
    assert abs(res.coefficients["(Intercept)"] - 1.0) < 0.5
    assert abs(res.coefficients["x0"] - 3.0) < 0.5


def test_wls_equal_weights_matches_ols():
    rng = np.random.default_rng(99)
    n = 100
    X = rng.standard_normal((n, 1))
    y = 2.0 + X[:, 0] + rng.standard_normal(n) * 0.5
    w = np.ones(n)
    from moirais.fn.olsrg import ols_regression
    ols = ols_regression(y, X)
    wls = wls_regression(y, X, w)
    np.testing.assert_allclose(
        list(ols.coefficients.values()),
        list(wls.coefficients.values()),
        atol=1e-10,
    )


def test_wls_negative_weights_raises():
    with pytest.raises(ValueError, match="positive"):
        wls_regression(np.ones(5), np.ones((5, 1)), np.array([-1, 1, 1, 1, 1]))


def test_wls_r_squared():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((100, 1))
    y = 5.0 * X[:, 0] + rng.standard_normal(100) * 0.1
    w = np.ones(100)
    res = wls_regression(y, X, w)
    assert res.r_squared > 0.95
