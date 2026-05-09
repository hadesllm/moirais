"""Tests for mdpval.mdp_value_iteration."""
import numpy as np
import pytest
from moirais.fn.mdpval import mdp_value_iteration


def test_mdpval_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    tol = 1e-6
    result = mdp_value_iteration(P, R, gamma, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mdpval_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    tol = 1e-6
    result = mdp_value_iteration(P, R, gamma, tol)
    assert isinstance(result, dict)
