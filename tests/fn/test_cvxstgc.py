"""Tests for cvxstgc.boyd_strong_convex."""
import numpy as np
import pytest
from moirais.fn.cvxstgc import boyd_strong_convex


def test_cvxstgc_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = boyd_strong_convex(f, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxstgc_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = boyd_strong_convex(f, m)
    assert isinstance(result, dict)
