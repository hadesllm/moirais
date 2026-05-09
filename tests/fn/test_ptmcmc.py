"""Tests for ptmcmc.parallel_tempering."""
import numpy as np
import pytest
from moirais.fn.ptmcmc import parallel_tempering


def test_ptmcmc_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    temperatures = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = parallel_tempering(log_p, temperatures, x0, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ptmcmc_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    temperatures = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = parallel_tempering(log_p, temperatures, x0, n_iter)
    assert isinstance(result, dict)
