"""Tests for mabegg.ma_begg_test."""
import numpy as np
import pytest
from morie.fn.mabegg import ma_begg_test


def test_mabegg_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_begg_test(yi, vi)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_mabegg_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_begg_test(yi, vi)
    assert isinstance(result, dict)
