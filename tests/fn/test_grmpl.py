"""Tests for grmpl.geron_max_pooling."""
import numpy as np
import pytest
from morie.fn.grmpl import geron_max_pooling


def test_grmpl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_max_pooling(X, k, stride)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmpl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_max_pooling(X, k, stride)
    assert isinstance(result, dict)
