"""Tests for moirais.fn.bcred -- Bayesian credible interval."""

import numpy as np
from moirais.fn.bcred import credible_interval


def test_returns_dict():
    result = credible_interval([1, 2, 3, 4, 5])
    assert isinstance(result, dict)
    assert "ci_lower" in result
    assert "ci_upper" in result


def test_95_ci():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 10000)
    result = credible_interval(samples, prob=0.95)
    assert abs(result["ci_lower"] - (-1.96)) < 0.2
    assert abs(result["ci_upper"] - 1.96) < 0.2


def test_narrower_with_less_prob():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 5000)
    ci95 = credible_interval(samples, prob=0.95)
    ci50 = credible_interval(samples, prob=0.50)
    w95 = ci95["ci_upper"] - ci95["ci_lower"]
    w50 = ci50["ci_upper"] - ci50["ci_lower"]
    assert w50 < w95


def test_ci_contains_median():
    rng = np.random.default_rng(42)
    samples = rng.normal(5, 2, 1000)
    result = credible_interval(samples)
    assert result["ci_lower"] < result["median"] < result["ci_upper"]


def test_empty():
    try:
        credible_interval([])
        assert False
    except ValueError:
        pass


def test_invalid_prob():
    try:
        credible_interval([1, 2, 3], prob=1.5)
        assert False
    except ValueError:
        pass
