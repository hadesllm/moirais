"""Tests for cvxqp.boyd_quadratic_program."""
import numpy as np
import pytest
from moirais.fn.cvxqp import boyd_quadratic_program


def test_cvxqp_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    h = 0.3
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_quadratic_program(P, q, G, h, A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxqp_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    h = 0.3
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_quadratic_program(P, q, G, h, A, b)
    assert isinstance(result, dict)
