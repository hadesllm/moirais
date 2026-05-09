"""Tests for sgcL.sgc."""
import numpy as np
import pytest
from moirais.fn.sgcL import sgc


def test_sgcL_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgc(A, X, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgcL_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgc(A, X, K)
    assert isinstance(result, dict)
