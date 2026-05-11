"""Tests for cvxstx.boyd_strict_convex."""
import numpy as np
import pytest
from morie.fn.cvxstx import boyd_strict_convex


def test_cvxstx_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_strict_convex(f)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_cvxstx_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_strict_convex(f)
    assert isinstance(result, dict)
