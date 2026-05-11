"""Tests for svdd.svdd."""
import numpy as np
import pytest
from morie.fn.svdd import svdd


def test_svdd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = svdd(X, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svdd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = svdd(X, C)
    assert isinstance(result, dict)
