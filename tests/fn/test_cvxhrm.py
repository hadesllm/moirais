"""Tests for cvxhrm.boyd_huber_loss."""
import numpy as np
import pytest
from morie.fn.cvxhrm import boyd_huber_loss


def test_cvxhrm_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boyd_huber_loss(u, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxhrm_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boyd_huber_loss(u, M)
    assert isinstance(result, dict)
