"""Tests for moirais.fn.clse — Clustered standard errors."""

import numpy as np
import pytest

from moirais.fn.clse import clustered_se


def test_clustered_se_larger_than_ols():
    rng = np.random.default_rng(42)
    n_cl = 50
    cl_size = 10
    y, X, cl = [], [], []
    for c in range(n_cl):
        u = rng.standard_normal() * 2.0
        for _ in range(cl_size):
            x = rng.standard_normal()
            y.append(1.0 + 2.0 * x + u + rng.standard_normal() * 0.5)
            X.append([x])
            cl.append(c)
    y, X, cl = np.array(y), np.array(X), np.array(cl)
    from moirais.fn.olsrg import ols_regression
    ols = ols_regression(y, X)
    cse = clustered_se(y, X, cl)
    assert cse.se["(Intercept)"] > ols.se["(Intercept)"]


def test_clustered_se_coefs_same_as_ols():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 1))
    y = 2.0 + 3.0 * X[:, 0] + rng.standard_normal(100)
    cl = np.repeat(np.arange(20), 5)
    from moirais.fn.olsrg import ols_regression
    ols = ols_regression(y, X)
    cse = clustered_se(y, X, cl)
    np.testing.assert_allclose(
        list(ols.coefficients.values()),
        list(cse.coefficients.values()),
        atol=1e-10,
    )


def test_n_clusters():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 1))
    y = rng.standard_normal(60)
    cl = np.repeat(np.arange(12), 5)
    res = clustered_se(y, X, cl)
    assert res.extra["n_clusters"] == 12
