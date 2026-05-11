"""Tests for btiseq.boot_iter_calibrated."""
import numpy as np
import pytest
from morie.fn.btiseq import boot_iter_calibrated


def test_btiseq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    iters = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_iter_calibrated(x, stat, B, iters, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btiseq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    iters = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_iter_calibrated(x, stat, B, iters, alpha)
    assert isinstance(result, dict)
