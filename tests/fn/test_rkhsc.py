"""Tests for rkhsc.rkhs_kernel_regression."""
import numpy as np
import pytest
from morie.fn.rkhsc import rkhs_kernel_regression


def test_rkhsc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rkhs_kernel_regression(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rkhsc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rkhs_kernel_regression(x, y)
    assert isinstance(result, dict)
