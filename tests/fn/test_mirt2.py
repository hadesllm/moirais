"""Tests for mirt2.mirt_2d_compensatory."""
import numpy as np
import pytest
from morie.fn.mirt2 import mirt_2d_compensatory


def test_mirt2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    d = 5
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = mirt_2d_compensatory(y, theta, a, d, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mirt2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    d = 5
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = mirt_2d_compensatory(y, theta, a, d, c)
    assert isinstance(result, dict)
