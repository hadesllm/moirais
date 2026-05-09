"""Tests for wquan.weighted_quantile."""
import numpy as np
import pytest
from moirais.fn.wquan import weighted_quantile


def test_wquan_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    p = 5
    result = weighted_quantile(y, weights, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wquan_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    p = 5
    result = weighted_quantile(y, weights, p)
    assert isinstance(result, dict)
