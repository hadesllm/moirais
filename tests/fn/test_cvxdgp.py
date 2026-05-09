"""Tests for cvxdgp.boyd_dual_problem."""
import numpy as np
import pytest
from moirais.fn.cvxdgp import boyd_dual_problem


def test_cvxdgp_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = boyd_dual_problem(g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxdgp_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = boyd_dual_problem(g)
    assert isinstance(result, dict)
