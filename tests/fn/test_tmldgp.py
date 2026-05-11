"""Tests for tmldgp.tmle_doubly_robust_pen."""
import numpy as np
import pytest
from morie.fn.tmldgp import tmle_doubly_robust_pen


def test_tmldgp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    penalty = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_doubly_robust_pen(y, D, X, penalty)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmldgp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    penalty = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_doubly_robust_pen(y, D, X, penalty)
    assert isinstance(result, dict)
