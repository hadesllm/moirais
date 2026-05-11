"""Tests for morie.fn.bfact -- Savage-Dickey Bayes factor."""

import numpy as np
from morie.fn.bfact import bayes_factor_savage_dickey


def test_returns_dict():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 500)
    result = bayes_factor_savage_dickey(samples)
    assert isinstance(result, dict)
    assert "bf01" in result
    assert "bf10" in result


def test_null_true_favours_null():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 0.5, 2000)
    result = bayes_factor_savage_dickey(
        samples, prior_mean=0, prior_sd=1, null_value=0
    )
    assert result["bf01"] > 1.0


def test_null_false_favours_alt():
    rng = np.random.default_rng(42)
    samples = rng.normal(5, 0.3, 2000)
    result = bayes_factor_savage_dickey(
        samples, prior_mean=0, prior_sd=1, null_value=0
    )
    assert result["bf10"] > 1.0


def test_bf01_bf10_reciprocal():
    rng = np.random.default_rng(42)
    samples = rng.normal(1, 1, 500)
    result = bayes_factor_savage_dickey(samples)
    np.testing.assert_allclose(result["bf01"] * result["bf10"], 1.0, atol=1e-6)


def test_evidence_category_present():
    rng = np.random.default_rng(42)
    result = bayes_factor_savage_dickey(rng.normal(0, 1, 500))
    assert isinstance(result["evidence_category"], str)


def test_too_few_samples():
    try:
        bayes_factor_savage_dickey([1.0, 2.0])
        assert False
    except ValueError:
        pass
