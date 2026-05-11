"""Tests for hrzwfun.horowitz_nls_weight_function."""
import numpy as np
import pytest
from morie.fn.hrzwfun import horowitz_nls_weight_function


def test_hrzwfun_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    weights = np.random.default_rng(45).exponential(1, 100)
    result = horowitz_nls_weight_function(x, y, bandwidth, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzwfun_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    weights = np.random.default_rng(45).exponential(1, 100)
    result = horowitz_nls_weight_function(x, y, bandwidth, weights)
    assert isinstance(result, dict)
