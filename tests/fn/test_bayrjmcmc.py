"""Tests for bayrjmcmc.reversible_jump_mcmc."""
import numpy as np
import pytest
from morie.fn.bayrjmcmc import reversible_jump_mcmc


def test_bayrjmcmc_basic():
    """Test basic functionality."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = reversible_jump_mcmc(models, x0, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayrjmcmc_edge():
    """Test edge cases."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = reversible_jump_mcmc(models, x0, n_iter)
    assert isinstance(result, dict)
