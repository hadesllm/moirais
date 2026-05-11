"""Tests for causbalt.causal_balance_test."""
import numpy as np
import pytest
from morie.fn.causbalt import causal_balance_test


def test_causbalt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    treat = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = causal_balance_test(X, treat, weights)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_causbalt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    treat = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = causal_balance_test(X, treat, weights)
    assert isinstance(result, dict)
