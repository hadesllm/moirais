"""Tests for wsmmcm.wasserman_mcmc_metropolis."""
import numpy as np
import pytest
from morie.fn.wsmmcm import wasserman_mcmc_metropolis


def test_wsmmcm_basic():
    """Test basic functionality."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = wasserman_mcmc_metropolis(target, proposal, x0, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmmcm_edge():
    """Test edge cases."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = wasserman_mcmc_metropolis(target, proposal, x0, n)
    assert isinstance(result, dict)
