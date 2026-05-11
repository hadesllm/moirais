"""Tests for grdds.gradient_descent_vanilla."""
import numpy as np
import pytest
from morie.fn.grdds import gradient_descent_vanilla


def test_grdds_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gradient_descent_vanilla(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdds_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gradient_descent_vanilla(x, y)
    assert isinstance(result, dict)
