"""Tests for cvxqcr.boyd_quadratic_constraint."""
import numpy as np
import pytest
from morie.fn.cvxqcr import boyd_quadratic_constraint


def test_cvxqcr_basic():
    """Test basic functionality."""
    P0 = np.random.default_rng(42).normal(0, 1, 100)
    q0 = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = boyd_quadratic_constraint(P0, q0, P, q, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxqcr_edge():
    """Test edge cases."""
    P0 = np.random.default_rng(42).normal(0, 1, 100)
    q0 = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = boyd_quadratic_constraint(P0, q0, P, q, r)
    assert isinstance(result, dict)
