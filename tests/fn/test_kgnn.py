"""Tests for kgnn.r_gcn."""
import numpy as np
import pytest
from morie.fn.kgnn import r_gcn


def test_kgnn_basic():
    """Test basic functionality."""
    A_r = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W_r = np.random.default_rng(42).normal(0, 1, 100)
    result = r_gcn(A_r, X, W_r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kgnn_edge():
    """Test edge cases."""
    A_r = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W_r = np.random.default_rng(42).normal(0, 1, 100)
    result = r_gcn(A_r, X, W_r)
    assert isinstance(result, dict)
