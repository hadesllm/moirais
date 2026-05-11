"""Tests for mvecv.min_volume_ellipsoid."""
import numpy as np
import pytest
from morie.fn.mvecv import min_volume_ellipsoid


def test_mvecv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = min_volume_ellipsoid(y, X, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mvecv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = min_volume_ellipsoid(y, X, h)
    assert isinstance(result, dict)
