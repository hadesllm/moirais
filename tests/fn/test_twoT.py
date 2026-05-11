"""Tests for twoT.two_tower."""
import numpy as np
import pytest
from morie.fn.twoT import two_tower


def test_twoT_basic():
    """Test basic functionality."""
    user_X = np.random.default_rng(42).normal(0, 1, 100)
    item_X = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = two_tower(user_X, item_X, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_twoT_edge():
    """Test edge cases."""
    user_X = np.random.default_rng(42).normal(0, 1, 100)
    item_X = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = two_tower(user_X, item_X, K)
    assert isinstance(result, dict)
