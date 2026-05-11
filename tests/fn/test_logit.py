"""Tests for morie.fn.logit — Logistic regression via IRLS."""

import numpy as np
import pytest

from morie.fn.logit import logistic_regression


@pytest.fixture()
def logit_data():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    prob = 1.0 / (1.0 + np.exp(-(1.0 + 2.0 * X[:, 0])))
    y = (rng.uniform(size=n) < prob).astype(float)
    return y, X


def test_coefficients_direction(logit_data):
    y, X = logit_data
    res = logistic_regression(y, X)
    assert res.coefficients["x0"] > 0


def test_intercept_plausible(logit_data):
    y, X = logit_data
    res = logistic_regression(y, X)
    assert abs(res.coefficients["(Intercept)"] - 1.0) < 1.0


def test_fitted_in_0_1(logit_data):
    y, X = logit_data
    res = logistic_regression(y, X)
    assert np.all(res.fitted >= 0)
    assert np.all(res.fitted <= 1)


def test_p_value_significant(logit_data):
    y, X = logit_data
    res = logistic_regression(y, X)
    assert res.p_values["x0"] < 0.05


def test_deviance_positive(logit_data):
    y, X = logit_data
    res = logistic_regression(y, X)
    assert res.extra["deviance"] > 0
    assert res.extra["aic"] > 0


def test_pseudo_r2_positive(logit_data):
    y, X = logit_data
    res = logistic_regression(y, X)
    assert res.r_squared > 0
