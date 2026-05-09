"""Tests for moirais.fn.gamma — Gamma GLM."""

import numpy as np
import pytest

from moirais.fn.gamma import gamma_glm


def test_gamma_glm_positive_coef():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    mu = np.exp(1.0 + 0.5 * X[:, 0])
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    res = gamma_glm(y, X)
    assert res.coefficients["x0"] > 0


def test_gamma_glm_negative_y_raises():
    with pytest.raises(ValueError, match="positive"):
        gamma_glm(np.array([-1.0, 2.0, 3.0]), np.ones((3, 1)))


def test_gamma_glm_fitted_positive():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = np.exp(0.5 + 0.3 * X[:, 0]) + rng.exponential(0.1, n)
    res = gamma_glm(y, X)
    assert np.all(res.fitted > 0)


def test_gamma_glm_deviance():
    rng = np.random.default_rng(42)
    n = 150
    X = rng.standard_normal((n, 1))
    y = rng.gamma(shape=3.0, scale=np.exp(0.5 * X[:, 0]) / 3.0)
    res = gamma_glm(y, X)
    assert res.extra["deviance"] > 0
    assert np.isfinite(res.extra["phi"])
