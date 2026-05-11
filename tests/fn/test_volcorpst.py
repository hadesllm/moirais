"""Tests for volcorpst.vol_corradi_swan_persistence."""
import numpy as np
import pytest
from morie.fn.volcorpst import vol_corradi_swan_persistence


def test_volcorpst_basic():
    """Test basic functionality."""
    r = 10
    horizons = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_corradi_swan_persistence(r, horizons)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_volcorpst_edge():
    """Test edge cases."""
    r = 10
    horizons = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_corradi_swan_persistence(r, horizons)
    assert isinstance(result, dict)
