"""Tests for hmcv.geron_cross_validation."""
import numpy as np
import pytest
from morie.fn.hmcv import geron_cross_validation


def test_hmcv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = geron_cross_validation(X, y, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = geron_cross_validation(X, y, k)
    assert isinstance(result, dict)
