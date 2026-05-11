"""Tests for morie.fn.nbglm — Negative binomial GLM."""

import numpy as np
import pytest

from morie.fn.nbglm import negbin_glm


def test_nbglm_positive_coef():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 1))
    mu = np.exp(0.5 + 1.0 * X[:, 0])
    y = rng.negative_binomial(n=5, p=5 / (5 + mu))
    res = negbin_glm(y.astype(float), X, alpha=0.2)
    assert res.coefficients["x0"] > 0


def test_nbglm_fitted_positive():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = rng.poisson(lam=np.exp(1.0 + 0.5 * X[:, 0]))
    res = negbin_glm(y.astype(float), X, alpha=0.5)
    assert np.all(res.fitted > 0)


def test_nbglm_aic_finite():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 1))
    y = rng.poisson(lam=np.exp(0.5 * X[:, 0])).astype(float)
    res = negbin_glm(y, X, alpha=1.0)
    assert np.isfinite(res.extra["aic"])
    assert np.isfinite(res.extra["log_likelihood"])


def test_nbglm_alpha_zero_like_poisson():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = rng.poisson(lam=np.exp(1.0 + 0.5 * X[:, 0])).astype(float)
    from morie.fn.pois import poisson_regression
    pois = poisson_regression(y, X)
    nb = negbin_glm(y, X, alpha=0.001)
    assert abs(nb.coefficients["x0"] - pois.extra["coefficients"]["x0"]) < 0.3
