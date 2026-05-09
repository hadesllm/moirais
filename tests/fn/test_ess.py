"""Tests for moirais.fn.ess -- MCMC effective sample size."""

import numpy as np
from moirais.fn.ess import effective_sample_size


def test_returns_dict():
    rng = np.random.default_rng(42)
    result = effective_sample_size(rng.normal(0, 1, 500))
    assert isinstance(result, dict)
    assert "ess" in result
    assert "n" in result
    assert "efficiency" in result


def test_iid_high_ess():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 1000)
    result = effective_sample_size(samples)
    assert result["ess"] > 500
    assert result["efficiency"] > 0.5


def test_correlated_low_ess():
    rng = np.random.default_rng(42)
    n = 1000
    samples = np.empty(n)
    samples[0] = 0.0
    for i in range(1, n):
        samples[i] = 0.99 * samples[i - 1] + rng.normal(0, 0.1)
    result = effective_sample_size(samples)
    assert result["ess"] < 200
    assert result["efficiency"] < 0.3


def test_constant_chain():
    result = effective_sample_size(np.ones(100))
    assert result["ess"] == 100.0


def test_efficiency_bounded():
    rng = np.random.default_rng(42)
    result = effective_sample_size(rng.normal(0, 1, 500))
    assert 0 < result["efficiency"] <= 1.0


def test_too_short():
    try:
        effective_sample_size([1, 2, 3])
        assert False
    except ValueError:
        pass
