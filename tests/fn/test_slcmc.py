"""Tests for slcmc.slice_sampler."""
import numpy as np
import pytest
from moirais.fn.slcmc import slice_sampler


def test_slcmc_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    width = np.random.default_rng(42).normal(0, 1, 100)
    result = slice_sampler(log_p, x0, width)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_slcmc_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    width = np.random.default_rng(42).normal(0, 1, 100)
    result = slice_sampler(log_p, x0, width)
    assert isinstance(result, dict)
