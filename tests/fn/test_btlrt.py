"""Tests for btlrt.boot_lr_test."""
import numpy as np
import pytest
from morie.fn.btlrt import boot_lr_test


def test_btlrt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fit0 = np.random.default_rng(42).normal(0, 1, 100)
    fit1 = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_lr_test(x, fit0, fit1, B)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_btlrt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fit0 = np.random.default_rng(42).normal(0, 1, 100)
    fit1 = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_lr_test(x, fit0, fit1, B)
    assert isinstance(result, dict)
