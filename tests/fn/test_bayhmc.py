"""Tests for bayhmc.hmc_dual_avg."""
import numpy as np
import pytest
from moirais.fn.bayhmc import hmc_dual_avg


def test_bayhmc_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    grad = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    target_accept = np.random.default_rng(42).normal(0, 1, 100)
    result = hmc_dual_avg(log_p, grad, x0, target_accept)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayhmc_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    grad = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    target_accept = np.random.default_rng(42).normal(0, 1, 100)
    result = hmc_dual_avg(log_p, grad, x0, target_accept)
    assert isinstance(result, dict)
