"""Tests for morie.fn.glsrg — GLS regression."""

import numpy as np
import pytest

from morie.fn.glsrg import gls_regression


def test_gls_identity_omega_matches_ols():
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 1))
    y = 2.0 + 3.0 * X[:, 0] + rng.standard_normal(n) * 0.5
    Omega = np.eye(n) * 0.25
    from morie.fn.olsrg import ols_regression
    ols = ols_regression(y, X)
    gls = gls_regression(y, X, Omega)
    np.testing.assert_allclose(
        list(ols.coefficients.values()),
        list(gls.coefficients.values()),
        atol=1e-8,
    )


def test_gls_with_ar1_correlation():
    rng = np.random.default_rng(77)
    n = 100
    rho = 0.7
    Omega = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Omega[i, j] = rho ** abs(i - j)

    L = np.linalg.cholesky(Omega)
    X = rng.standard_normal((n, 1))
    y = 1.0 + 2.0 * X[:, 0] + L @ rng.standard_normal(n)
    res = gls_regression(y, X, Omega)
    assert abs(res.coefficients["x0"] - 2.0) < 1.0
    assert res.n == n


def test_gls_non_pd_raises():
    X = np.ones((5, 1))
    y = np.arange(5, dtype=float)
    Omega = -np.eye(5)
    with pytest.raises(ValueError, match="positive definite"):
        gls_regression(y, X, Omega)
