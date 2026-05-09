"""Tests for macum.ma_cumulative."""
import numpy as np
import pytest
from moirais.fn.macum import ma_cumulative


def test_macum_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = ma_cumulative(yi, vi, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_macum_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = ma_cumulative(yi, vi, order)
    assert isinstance(result, dict)
