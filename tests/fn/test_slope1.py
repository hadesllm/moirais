"""Tests for slope1.slope_one."""
import numpy as np
import pytest
from morie.fn.slope1 import slope_one


def test_slope1_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = slope_one(R, u, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_slope1_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = slope_one(R, u, i)
    assert isinstance(result, dict)
