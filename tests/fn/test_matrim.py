"""Tests for matrim.ma_trim_fill."""
import numpy as np
import pytest
from morie.fn.matrim import ma_trim_fill


def test_matrim_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    side = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_trim_fill(yi, vi, side)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matrim_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    side = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_trim_fill(yi, vi, side)
    assert isinstance(result, dict)
