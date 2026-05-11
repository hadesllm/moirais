"""Tests for morie.fn.betag — Beta regression."""

import numpy as np
import pytest

from morie.fn.betag import beta_regression


def test_beta_regression_recovers_direction():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 1))
    mu = 1 / (1 + np.exp(-(0.5 + 1.0 * X[:, 0])))
    phi = 20.0
    a = mu * phi
    b = (1 - mu) * phi
    y = rng.beta(a, b)
    y = np.clip(y, 0.001, 0.999)
    res = beta_regression(y, X)
    assert res.coefficients["x0"] > 0


def test_beta_regression_out_of_range_raises():
    with pytest.raises(ValueError, match="\\(0, 1\\)"):
        beta_regression(np.array([0.0, 0.5, 1.0]), np.ones((3, 1)))


def test_beta_regression_fitted_in_0_1():
    rng = np.random.default_rng(7)
    n = 100
    X = rng.standard_normal((n, 1))
    y = rng.beta(2, 5, size=n)
    y = np.clip(y, 0.001, 0.999)
    res = beta_regression(y, X)
    assert np.all(res.fitted > 0)
    assert np.all(res.fitted < 1)


def test_beta_regression_phi_positive():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = rng.beta(5, 5, size=n)
    y = np.clip(y, 0.001, 0.999)
    res = beta_regression(y, X)
    assert res.extra["phi"] > 0
