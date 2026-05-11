"""Tests for dpqua.dp_quantile."""
import numpy as np
import pytest
from morie.fn.dpqua import dp_quantile


def test_dpqua_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_quantile(x, q, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpqua_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_quantile(x, q, epsilon)
    assert isinstance(result, dict)
