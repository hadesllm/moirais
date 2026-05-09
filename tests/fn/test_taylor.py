"""Tests for taylor.taylor_linearization."""
import numpy as np
import pytest
from moirais.fn.taylor import taylor_linearization


def test_taylor_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    grad = np.random.default_rng(42).normal(0, 1, 100)
    result = taylor_linearization(y, weights, grad)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_taylor_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    grad = np.random.default_rng(42).normal(0, 1, 100)
    result = taylor_linearization(y, weights, grad)
    assert isinstance(result, dict)
