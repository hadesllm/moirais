"""Tests for btatest.boot_anderson_test."""
import numpy as np
import pytest
from morie.fn.btatest import boot_anderson_test


def test_btatest_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fit = np.random.default_rng(42).normal(0, 1, 100)
    rvs_fn = (lambda v: v)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_anderson_test(x, fit, rvs_fn, B)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_btatest_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fit = np.random.default_rng(42).normal(0, 1, 100)
    rvs_fn = (lambda v: v)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_anderson_test(x, fit, rvs_fn, B)
    assert isinstance(result, dict)
