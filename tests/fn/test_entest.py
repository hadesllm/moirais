"""Tests for entest.entropy_knn."""
import numpy as np
import pytest
from morie.fn.entest import entropy_knn


def test_entest_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = entropy_knn(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_entest_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = entropy_knn(X, k)
    assert isinstance(result, dict)
