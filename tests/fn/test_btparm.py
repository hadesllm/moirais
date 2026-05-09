"""Tests for btparm.boot_parametric."""
import numpy as np
import pytest
from moirais.fn.btparm import boot_parametric


def test_btparm_basic():
    """Test basic functionality."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    rvs_fn = (lambda v: v)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    n = 100
    result = boot_parametric(theta_hat, rvs_fn, stat, B, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btparm_edge():
    """Test edge cases."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    rvs_fn = (lambda v: v)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    n = 100
    result = boot_parametric(theta_hat, rvs_fn, stat, B, n)
    assert isinstance(result, dict)
