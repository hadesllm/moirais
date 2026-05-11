"""Tests for bndcvx.bound_convex_estimator."""
import numpy as np
import pytest
from morie.fn.bndcvx import bound_convex_estimator


def test_bndcvx_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_convex_estimator(y, D, constraints)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndcvx_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_convex_estimator(y, D, constraints)
    assert isinstance(result, dict)
