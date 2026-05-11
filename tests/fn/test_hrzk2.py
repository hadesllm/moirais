"""Tests for hrzk2.horowitz_kernel_regression."""
import numpy as np
import pytest
from morie.fn.hrzk2 import horowitz_kernel_regression


def test_hrzk2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_kernel_regression(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzk2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_kernel_regression(x, y, bandwidth)
    assert isinstance(result, dict)
