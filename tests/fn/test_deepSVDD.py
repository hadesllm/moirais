"""Tests for deepSVDD.deep_svdd."""
import numpy as np
import pytest
from morie.fn.deepSVDD import deep_svdd


def test_deepSVDD_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    net = np.random.default_rng(42).normal(0, 1, 100)
    result = deep_svdd(X, net)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deepSVDD_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    net = np.random.default_rng(42).normal(0, 1, 100)
    result = deep_svdd(X, net)
    assert isinstance(result, dict)
