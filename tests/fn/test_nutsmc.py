"""Tests for nutsmc.nuts_sampler."""
import numpy as np
import pytest
from moirais.fn.nutsmc import nuts_sampler


def test_nutsmc_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    grad_log_p = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = nuts_sampler(log_p, grad_log_p, x0, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nutsmc_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    grad_log_p = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = nuts_sampler(log_p, grad_log_p, x0, n_iter)
    assert isinstance(result, dict)
