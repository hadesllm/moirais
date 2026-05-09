"""Tests for cvxcvv.boyd_cvxlin_complement."""
import numpy as np
import pytest
from moirais.fn.cvxcvv import boyd_cvxlin_complement


def test_cvxcvv_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_cvxlin_complement(A, B, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxcvv_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_cvxlin_complement(A, B, C)
    assert isinstance(result, dict)
