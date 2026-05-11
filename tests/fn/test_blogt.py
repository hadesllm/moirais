"""Tests for morie.fn.blogt -- Bayesian logistic regression."""

import numpy as np
import pytest
from morie.fn.blogt import bayesian_logistic


@pytest.fixture()
def data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    p = 1 / (1 + np.exp(-(X @ [1.5, -0.5])))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y


def test_returns_dict(data):
    result = bayesian_logistic(data[0], data[1])
    assert isinstance(result, dict)
    assert "beta_map" in result


def test_converges(data):
    result = bayesian_logistic(data[0], data[1])
    assert result["converged"]


def test_beta_positive_first(data):
    result = bayesian_logistic(data[0], data[1])
    assert result["beta_map"][0] > 0
