"""Tests for cvxlse.boyd_lse."""
import numpy as np
import pytest
from morie.fn.cvxlse import boyd_lse


def test_cvxlse_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_lse(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxlse_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_lse(x)
    assert isinstance(result, dict)
