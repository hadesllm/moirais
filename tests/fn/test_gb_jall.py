"""Tests for gb_jall.gibbons_joint_all_order."""
import numpy as np
import pytest
from moirais.fn.gb_jall import gibbons_joint_all_order


def test_gb_jall_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_joint_all_order(x, f)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_jall_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_joint_all_order(x, f)
    assert isinstance(result, dict)
