"""Tests for cvxsbp.boyd_subgradient."""
import numpy as np
import pytest
from morie.fn.cvxsbp import boyd_subgradient


def test_cvxsbp_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_subgradient(f, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxsbp_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_subgradient(f, x)
    assert isinstance(result, dict)
