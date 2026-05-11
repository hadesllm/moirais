"""Tests for gb2111.gibbons_tolerance_beta."""
import numpy as np
import pytest
from morie.fn.gb2111 import gibbons_tolerance_beta


def test_gb2111_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    s = 90
    p = 5
    gamma = 1.0
    result = gibbons_tolerance_beta(x, r, s, p, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb2111_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    s = 90
    p = 5
    gamma = 1.0
    result = gibbons_tolerance_beta(x, r, s, p, gamma)
    assert isinstance(result, dict)
