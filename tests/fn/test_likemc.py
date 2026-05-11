"""Tests for likemc.likelihood_mcmc_epi."""
import numpy as np
import pytest
from morie.fn.likemc import likelihood_mcmc_epi


def test_likemc_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = likelihood_mcmc_epi(model, data, priors, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_likemc_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = likelihood_mcmc_epi(model, data, priors, n_iter)
    assert isinstance(result, dict)
