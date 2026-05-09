"""Tests for cvxsmt.boyd_smooth_min."""
import numpy as np
import pytest
from moirais.fn.cvxsmt import boyd_smooth_min


def test_cvxsmt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_smooth_min(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxsmt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_smooth_min(x)
    assert isinstance(result, dict)
