"""Tests for ardlmd.ardl_bounds_test."""
import numpy as np
import pytest
from moirais.fn.ardlmd import ardl_bounds_test


def test_ardlmd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = ardl_bounds_test(y, X, p, q)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ardlmd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = ardl_bounds_test(y, X, p, q)
    assert isinstance(result, dict)
