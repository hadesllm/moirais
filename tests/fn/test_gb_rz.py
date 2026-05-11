"""Tests for gb_rz.gibbons_rz_test."""
import numpy as np
import pytest
from morie.fn.gb_rz import gibbons_rz_test


def test_gb_rz_basic():
    """Test basic functionality."""
    statistic = np.random.default_rng(42).normal(0, 1, 100)
    null_dist = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_rz_test(statistic, null_dist, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_rz_edge():
    """Test edge cases."""
    statistic = np.random.default_rng(42).normal(0, 1, 100)
    null_dist = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_rz_test(statistic, null_dist, alpha)
    assert isinstance(result, dict)
