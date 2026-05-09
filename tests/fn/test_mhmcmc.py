"""Tests for mhmcmc.metropolis_hastings."""
import numpy as np
import pytest
from moirais.fn.mhmcmc import metropolis_hastings


def test_mhmcmc_basic():
    """Test basic functionality."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = metropolis_hastings(target, proposal, x0, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mhmcmc_edge():
    """Test edge cases."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = metropolis_hastings(target, proposal, x0, n_iter)
    assert isinstance(result, dict)
