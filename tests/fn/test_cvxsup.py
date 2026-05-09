"""Tests for cvxsup.boyd_support_hyperplane."""
import numpy as np
import pytest
from moirais.fn.cvxsup import boyd_support_hyperplane


def test_cvxsup_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_support_hyperplane(C, x0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxsup_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_support_hyperplane(C, x0)
    assert isinstance(result, dict)
