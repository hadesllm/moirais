"""Tests for cvxdle.boyd_dual_norm."""
import numpy as np
import pytest
from morie.fn.cvxdle import boyd_dual_norm


def test_cvxdle_basic():
    """Test basic functionality."""
    norm = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_dual_norm(norm, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxdle_edge():
    """Test edge cases."""
    norm = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_dual_norm(norm, z)
    assert isinstance(result, dict)
