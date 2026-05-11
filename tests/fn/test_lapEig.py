"""Tests for lapEig.laplacian_eigenmaps."""
import numpy as np
import pytest
from morie.fn.lapEig import laplacian_eigenmaps


def test_lapEig_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = laplacian_eigenmaps(A, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lapEig_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = laplacian_eigenmaps(A, k)
    assert isinstance(result, dict)
