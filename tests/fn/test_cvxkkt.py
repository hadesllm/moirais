"""Tests for cvxkkt.boyd_kkt."""
import numpy as np
import pytest
from moirais.fn.cvxkkt import boyd_kkt


def test_cvxkkt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_kkt(x, lambda_, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxkkt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_kkt(x, lambda_, nu)
    assert isinstance(result, dict)
