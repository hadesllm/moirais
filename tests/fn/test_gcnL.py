"""Tests for gcnL.gcn."""
import numpy as np
import pytest
from moirais.fn.gcnL import gcn


def test_gcnL_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = gcn(A, X, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gcnL_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = gcn(A, X, W)
    assert isinstance(result, dict)
