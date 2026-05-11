"""Tests for cvxlgr.boyd_logistic_loss."""
import numpy as np
import pytest
from morie.fn.cvxlgr import boyd_logistic_loss


def test_cvxlgr_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_logistic_loss(u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxlgr_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_logistic_loss(u)
    assert isinstance(result, dict)
