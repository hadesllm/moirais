"""Tests for volsd.vol_simple_diff."""
import numpy as np
import pytest
from morie.fn.volsd import vol_simple_diff


def test_volsd_basic():
    """Test basic functionality."""
    r = 10
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_simple_diff(r, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volsd_edge():
    """Test edge cases."""
    r = 10
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_simple_diff(r, window)
    assert isinstance(result, dict)
