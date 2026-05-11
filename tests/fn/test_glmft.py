"""Tests for glmft — Generalized linear model fitting."""
import numpy as np
import pytest
from morie.fn.glmft import glm_fit
from morie.fn._containers import RegressionResult


def test_glmft_gaussian(rng):
    n = 200
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.5 + rng.standard_normal(n) * 0.3
    result = glm_fit(y, x.reshape(-1, 1), family="gaussian")
    assert isinstance(result, RegressionResult)
    assert abs(result.coefficients["x0"] - 2.0) < 0.3
    assert abs(result.coefficients["intercept"] - 0.5) < 0.3


def test_glmft_poisson(rng):
    n = 300
    x = rng.standard_normal(n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(mu)
    result = glm_fit(y, x.reshape(-1, 1), family="poisson")
    assert result.extra["deviance"] > 0
    assert result.extra["aic"] > 0


def test_glmft_binomial(rng):
    n = 300
    x = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.5 + x)))
    y = rng.binomial(1, prob)
    result = glm_fit(y, x.reshape(-1, 1), family="binomial")
    assert result.coefficients["x0"] > 0


def test_glmft_invalid_family():
    with pytest.raises(ValueError, match="Unknown family"):
        glm_fit(np.array([1, 2, 3]), np.array([[1], [2], [3]]), family="bogus")
