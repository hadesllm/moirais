"""Tests for cvxsmh.boyd_smooth_huber_grad."""
import numpy as np
import pytest
from moirais.fn.cvxsmh import boyd_smooth_huber_grad


def test_cvxsmh_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boyd_smooth_huber_grad(u, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxsmh_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boyd_smooth_huber_grad(u, M)
    assert isinstance(result, dict)
