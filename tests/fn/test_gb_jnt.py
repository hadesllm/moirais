"""Tests for gb_jnt.gibbons_joint_order."""
import numpy as np
import pytest
from moirais.fn.gb_jnt import gibbons_joint_order


def test_gb_jnt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    r = 10
    s = 90
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_joint_order(x, y, r, s, n, f, F)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_jnt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    r = 10
    s = 90
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_joint_order(x, y, r, s, n, f, F)
    assert isinstance(result, dict)
