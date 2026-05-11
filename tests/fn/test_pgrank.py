"""Tests for pgrank.pagerank."""
import numpy as np
import pytest
from morie.fn.pgrank import pagerank


def test_pgrank_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    d = 5
    tol = 1e-6
    result = pagerank(y, A, d, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pgrank_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    d = 5
    tol = 1e-6
    result = pagerank(y, A, d, tol)
    assert isinstance(result, dict)
