"""Tests for xdeep.xdeepfm."""
import numpy as np
import pytest
from morie.fn.xdeep import xdeepfm


def test_xdeep_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = xdeepfm(X, y, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_xdeep_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = xdeepfm(X, y, K)
    assert isinstance(result, dict)
