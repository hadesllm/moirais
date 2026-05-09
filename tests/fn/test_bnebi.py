"""Tests for moirais.fn.bnebi -- Bayesian negative binomial."""

from moirais.fn.bnebi import bayesian_negbinom


def test_returns_dict():
    result = bayesian_negbinom([3, 5, 2, 7])
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_posterior_mean_in_range():
    result = bayesian_negbinom([3, 5, 2, 7])
    assert 0 < result["posterior_mean"] < 1


def test_ci_contains_mean():
    result = bayesian_negbinom([10, 15, 12, 8, 11])
    assert result["ci_lower"] < result["posterior_mean"] < result["ci_upper"]
