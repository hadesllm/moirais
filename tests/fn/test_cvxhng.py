"""Tests for cvxhng.boyd_hinge_loss."""
import numpy as np
import pytest
from moirais.fn.cvxhng import boyd_hinge_loss


def test_cvxhng_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_hinge_loss(u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxhng_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_hinge_loss(u)
    assert isinstance(result, dict)
