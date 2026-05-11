"""Tests for cvxqpdl.boyd_qp_dual."""
import numpy as np
import pytest
from morie.fn.cvxqpdl import boyd_qp_dual


def test_cvxqpdl_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    h = 0.3
    result = boyd_qp_dual(P, q, G, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxqpdl_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    h = 0.3
    result = boyd_qp_dual(P, q, G, h)
    assert isinstance(result, dict)
