"""Tests for jkest.jackknife_estimator."""
import numpy as np
import pytest
from morie.fn.jkest import jackknife_estimator


def test_jkest_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = jackknife_estimator(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_jkest_edge():
    """Test edge cases."""
    result = jackknife_estimator(np.array([42.0]))
    assert result['n'] == 1
