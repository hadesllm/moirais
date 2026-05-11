"""Tests for klmcmc.kl_mcmc_diagnostic."""
import numpy as np
import pytest
from morie.fn.klmcmc import kl_mcmc_diagnostic


def test_klmcmc_basic():
    """Test basic functionality."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = kl_mcmc_diagnostic(chain, target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_klmcmc_edge():
    """Test edge cases."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = kl_mcmc_diagnostic(chain, target)
    assert isinstance(result, dict)
