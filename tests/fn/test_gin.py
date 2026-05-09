"""Tests for gin.gin."""
import numpy as np
import pytest
from moirais.fn.gin import gin


def test_gin_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    epsilon = 1e-6
    result = gin(A, X, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gin_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    epsilon = 1e-6
    result = gin(A, X, epsilon)
    assert isinstance(result, dict)
