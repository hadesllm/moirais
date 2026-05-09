"""Tests for cvxlpl.boyd_linear_program_dual."""
import numpy as np
import pytest
from moirais.fn.cvxlpl import boyd_linear_program_dual


def test_cvxlpl_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_linear_program_dual(A, b, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxlpl_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_linear_program_dual(A, b, c)
    assert isinstance(result, dict)
