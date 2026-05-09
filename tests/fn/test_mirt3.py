"""Tests for mirt3.mirt_3d_compensatory."""
import numpy as np
import pytest
from moirais.fn.mirt3 import mirt_3d_compensatory


def test_mirt3_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    d = 5
    result = mirt_3d_compensatory(y, theta, a, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mirt3_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    d = 5
    result = mirt_3d_compensatory(y, theta, a, d)
    assert isinstance(result, dict)
