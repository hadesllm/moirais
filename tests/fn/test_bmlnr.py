"""Tests for morie.fn.bmlnr -- Bayesian linear regression."""

import numpy as np
import pytest
from morie.fn.bmlnr import bayesian_linear_regression


@pytest.fixture()
def data():
    rng = np.random.default_rng(42)
    n, p = 100, 2
    X = rng.standard_normal((n, p))
    y = X @ [2.0, -1.0] + rng.standard_normal(n) * 0.5
    return X, y


def test_returns_dict(data):
    result = bayesian_linear_regression(data[0], data[1])
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_posterior_mean_close(data):
    result = bayesian_linear_regression(data[0], data[1])
    assert abs(result["posterior_mean"][0] - 2.0) < 1.0
    assert abs(result["posterior_mean"][1] - (-1.0)) < 1.0


def test_ci_contains_truth(data):
    result = bayesian_linear_regression(data[0], data[1])
    assert result["ci_lower"][0] < 2.0 < result["ci_upper"][0]
