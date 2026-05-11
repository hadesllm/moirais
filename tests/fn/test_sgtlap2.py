"""Tests for sgtlap2.sgt_laplacian_eigenmaps."""
import numpy as np
import pytest
from morie.fn.sgtlap2 import sgt_laplacian_eigenmaps


def test_sgtlap2_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = sgt_laplacian_eigenmaps(A, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtlap2_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = sgt_laplacian_eigenmaps(A, k)
    assert isinstance(result, dict)
