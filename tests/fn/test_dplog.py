"""Tests for dplog.dp_logistic."""
import numpy as np
import pytest
from moirais.fn.dplog import dp_logistic


def test_dplog_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilon = 1e-6
    method = 'auto'
    result = dp_logistic(X, y, epsilon, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dplog_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilon = 1e-6
    method = 'auto'
    result = dp_logistic(X, y, epsilon, method)
    assert isinstance(result, dict)
