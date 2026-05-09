"""Tests for moirais.fn.bpois -- Bayesian Poisson model."""

import numpy as np
from moirais.fn.bpois import bayesian_poisson


def test_returns_dict():
    result = bayesian_poisson(10, 5.0)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_conjugate_update():
    result = bayesian_poisson(10, 5.0, prior_a=1, prior_b=1)
    np.testing.assert_allclose(result["posterior_a"], 11.0)
    np.testing.assert_allclose(result["posterior_b"], 6.0)
    np.testing.assert_allclose(result["posterior_mean"], 11.0 / 6.0, atol=1e-10)


def test_ci_contains_mean():
    result = bayesian_poisson(20, 10.0)
    assert result["ci_lower"] < result["posterior_mean"] < result["ci_upper"]


def test_array_counts():
    result = bayesian_poisson([3, 4, 5], 10.0, prior_a=1, prior_b=1)
    np.testing.assert_allclose(result["posterior_a"], 13.0)


def test_variance_formula():
    result = bayesian_poisson(10, 5.0, prior_a=2, prior_b=1)
    a, b = result["posterior_a"], result["posterior_b"]
    np.testing.assert_allclose(result["posterior_var"], a / b ** 2, atol=1e-10)


def test_invalid_exposure():
    try:
        bayesian_poisson(10, -1)
        assert False
    except ValueError:
        pass
