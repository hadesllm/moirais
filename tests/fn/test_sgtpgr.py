"""Tests for sgtpgr.sgt_pagerank_power."""
import numpy as np
import pytest
from moirais.fn.sgtpgr import sgt_pagerank_power


def test_sgtpgr_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    d = 5
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = sgt_pagerank_power(A, d, max_iter, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtpgr_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    d = 5
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = sgt_pagerank_power(A, d, max_iter, tol)
    assert isinstance(result, dict)
