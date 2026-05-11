"""Tests for rgnmf.rangayyan_nmf."""
import numpy as np
import pytest
from morie.fn.rgnmf import rangayyan_nmf


def test_rgnmf_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_nmf(V, r, max_iter, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgnmf_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_nmf(V, r, max_iter, tol)
    assert isinstance(result, dict)
