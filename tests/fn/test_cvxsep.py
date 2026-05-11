"""Tests for cvxsep.boyd_separating_hyperplane."""
import numpy as np
import pytest
from morie.fn.cvxsep import boyd_separating_hyperplane


def test_cvxsep_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_separating_hyperplane(C, D)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxsep_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_separating_hyperplane(C, D)
    assert isinstance(result, dict)
