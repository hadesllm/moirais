"""Tests for sgtrwl.sgt_random_walk_laplacian."""
import numpy as np
import pytest
from moirais.fn.sgtrwl import sgt_random_walk_laplacian


def test_sgtrwl_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_random_walk_laplacian(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtrwl_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_random_walk_laplacian(A)
    assert isinstance(result, dict)
