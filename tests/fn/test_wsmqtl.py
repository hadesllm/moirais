"""Tests for wsmqtl.wasserman_empirical_quantile."""
import numpy as np
import pytest
from morie.fn.wsmqtl import wasserman_empirical_quantile


def test_wsmqtl_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = wasserman_empirical_quantile(data, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmqtl_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = wasserman_empirical_quantile(data, p)
    assert isinstance(result, dict)
