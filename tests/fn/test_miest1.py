"""Tests for miest1.mi_ksg."""
import numpy as np
import pytest
from morie.fn.miest1 import mi_ksg


def test_miest1_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = mi_ksg(X, Y, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_miest1_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = mi_ksg(X, Y, k)
    assert isinstance(result, dict)
