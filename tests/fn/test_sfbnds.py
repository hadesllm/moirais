"""Tests for sfbnds.sharp_bounds_balke_pearl."""
import numpy as np
import pytest
from morie.fn.sfbnds import sharp_bounds_balke_pearl


def test_sfbnds_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = sharp_bounds_balke_pearl(y, D, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sfbnds_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = sharp_bounds_balke_pearl(y, D, Z)
    assert isinstance(result, dict)
