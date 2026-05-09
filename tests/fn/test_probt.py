"""Tests for moirais.fn.probt — Probit regression."""

import numpy as np
import pytest
from scipy import stats

from moirais.fn.probt import probit_regression


def test_probit_positive_coef():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    prob = stats.norm.cdf(0.5 + 1.5 * X[:, 0])
    y = (rng.uniform(size=n) < prob).astype(float)
    res = probit_regression(y, X)
    assert res.coefficients["x0"] > 0.5


def test_probit_fitted_in_0_1():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = (X[:, 0] > 0).astype(float)
    res = probit_regression(y, X)
    assert np.all(res.fitted >= 0)
    assert np.all(res.fitted <= 1)


def test_probit_p_value():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    prob = stats.norm.cdf(1.0 * X[:, 0])
    y = (rng.uniform(size=n) < prob).astype(float)
    res = probit_regression(y, X)
    assert res.p_values["x0"] < 0.05


def test_probit_deviance():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    res = probit_regression(y, X)
    assert res.extra["deviance"] > 0
    assert np.isfinite(res.extra["aic"])
