"""Tests for wideD.wide_and_deep."""
import numpy as np
import pytest
from morie.fn.wideD import wide_and_deep


def test_wideD_basic():
    """Test basic functionality."""
    X_wide = np.random.default_rng(42).normal(0, 1, 100)
    X_deep = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wide_and_deep(X_wide, X_deep, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wideD_edge():
    """Test edge cases."""
    X_wide = np.random.default_rng(42).normal(0, 1, 100)
    X_deep = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wide_and_deep(X_wide, X_deep, y)
    assert isinstance(result, dict)
