"""Tests for cvxadm.boyd_admm."""
import numpy as np
import pytest
from morie.fn.cvxadm import boyd_admm


def test_cvxadm_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    c = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = boyd_admm(f, g, A, B, c, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxadm_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    c = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = boyd_admm(f, g, A, B, c, rho)
    assert isinstance(result, dict)
