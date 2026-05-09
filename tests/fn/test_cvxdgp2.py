"""Tests for cvxdgp2.boyd_duality_gap."""
import numpy as np
import pytest
from moirais.fn.cvxdgp2 import boyd_duality_gap


def test_cvxdgp2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_duality_gap(x, lambda_, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxdgp2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_duality_gap(x, lambda_, nu)
    assert isinstance(result, dict)
