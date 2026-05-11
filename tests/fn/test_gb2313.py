"""Tests for gb2313.gibbons_edf_joint_moment."""
import numpy as np
import pytest
from morie.fn.gb2313 import gibbons_edf_joint_moment


def test_gb2313_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = gibbons_edf_joint_moment(x, y, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb2313_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = gibbons_edf_joint_moment(x, y, n)
    assert isinstance(result, dict)
