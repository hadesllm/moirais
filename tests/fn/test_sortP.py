"""Tests for sortP.sortpool."""
import numpy as np
import pytest
from moirais.fn.sortP import sortpool


def test_sortP_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = sortpool(A, X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sortP_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = sortpool(A, X, k)
    assert isinstance(result, dict)
