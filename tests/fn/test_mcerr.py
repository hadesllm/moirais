"""Tests for morie.fn.mcerr -- MCMC standard error."""

import numpy as np
from morie.fn.mcerr import mcmc_se


def test_returns_dict():
    samples = np.random.default_rng(42).standard_normal(500)
    result = mcmc_se(samples)
    assert isinstance(result, dict)
    assert "mcse" in result


def test_mcse_positive():
    samples = np.random.default_rng(42).standard_normal(500)
    result = mcmc_se(samples)
    assert result["mcse"] > 0


def test_mcse_smaller_than_sd():
    samples = np.random.default_rng(42).standard_normal(1000)
    result = mcmc_se(samples)
    assert result["mcse"] < result["sd"]
