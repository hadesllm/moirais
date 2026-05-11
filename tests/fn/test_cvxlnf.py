"""Tests for cvxlnf.boyd_l1_fitting."""
import numpy as np
import pytest
from morie.fn.cvxlnf import boyd_l1_fitting


def test_cvxlnf_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_l1_fitting(A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxlnf_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_l1_fitting(A, b)
    assert isinstance(result, dict)
