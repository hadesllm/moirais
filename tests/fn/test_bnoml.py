"""Tests for moirais.fn.bnoml -- Bayesian binomial."""

from moirais.fn.bnoml import bayesian_binomial


def test_returns_dict():
    result = bayesian_binomial(7, 10)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_posterior_mean_in_range():
    result = bayesian_binomial(7, 10)
    assert 0 < result["posterior_mean"] < 1


def test_ci_contains_mean():
    result = bayesian_binomial(50, 100)
    assert result["ci_lower"] < result["posterior_mean"] < result["ci_upper"]


def test_strong_prior():
    result = bayesian_binomial(1, 2, prior_a=100, prior_b=100)
    assert abs(result["posterior_mean"] - 0.5) < 0.1
