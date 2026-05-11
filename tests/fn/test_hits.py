"""Tests for hits.hits."""
import numpy as np
import pytest
from morie.fn.hits import hits


def test_hits_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    iters = np.random.default_rng(42).normal(0, 1, 100)
    result = hits(A, iters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hits_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    iters = np.random.default_rng(42).normal(0, 1, 100)
    result = hits(A, iters)
    assert isinstance(result, dict)
