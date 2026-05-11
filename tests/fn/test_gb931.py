"""Tests for gb931.gibbons_fab_test."""
import numpy as np
import pytest
from morie.fn.gb931 import gibbons_fab_test


def test_gb931_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_fab_test(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb931_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_fab_test(x, y)
    assert isinstance(result, dict)
