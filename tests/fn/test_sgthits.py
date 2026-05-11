"""Tests for sgthits.sgt_hits_kleinberg."""
import numpy as np
import pytest
from morie.fn.sgthits import sgt_hits_kleinberg


def test_sgthits_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = sgt_hits_kleinberg(A, max_iter, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgthits_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = sgt_hits_kleinberg(A, max_iter, tol)
    assert isinstance(result, dict)
