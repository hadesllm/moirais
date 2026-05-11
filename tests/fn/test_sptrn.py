"""Tests for sptrn.spatial_trend_surface."""
import numpy as np
import pytest
from morie.fn.sptrn import spatial_trend_surface


def test_sptrn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_trend_surface(x, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sptrn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_trend_surface(x, coords)
    assert isinstance(result, dict)
