"""Tests for morie.fn.bfsdm -- Bayes factor Savage-Dickey."""

import numpy as np
from morie.fn.bfsdm import bayes_factor_savage_dickey


def test_returns_dict():
    samples = np.random.default_rng(42).normal(0, 1, 5000)
    result = bayes_factor_savage_dickey(samples)
    assert isinstance(result, dict)
    assert "bf01" in result


def test_bf_near_one_for_null():
    samples = np.random.default_rng(42).normal(0, 1, 10000)
    result = bayes_factor_savage_dickey(samples, null_value=0, prior_sd=1)
    assert result["bf01"] > 0


def test_evidence_against_null():
    samples = np.random.default_rng(42).normal(5, 0.5, 5000)
    result = bayes_factor_savage_dickey(samples, null_value=0, prior_sd=1)
    assert result["bf10"] > result["bf01"]
