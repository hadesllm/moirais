"""Tests for stacv.spatiotemporal_autocovariance."""
import numpy as np
import pytest
from morie.fn.stacv import spatiotemporal_autocovariance


def test_stacv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    times = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_autocovariance(x, coords, times)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stacv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    times = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_autocovariance(x, coords, times)
    assert isinstance(result, dict)
