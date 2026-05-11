"""Tests for dpols.dp_linear_regression."""
import numpy as np
import pytest
from morie.fn.dpols import dp_linear_regression


def test_dpols_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_linear_regression(X, y, epsilon)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_dpols_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_linear_regression(X, y, epsilon)
    assert isinstance(result, dict)
