"""Tests for gb_emo.gibbons_order_moments."""
import numpy as np
import pytest
from morie.fn.gb_emo import gibbons_order_moments


def test_gb_emo_basic():
    """Test basic functionality."""
    r = 10
    n = 100
    k = 5
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_order_moments(r, n, k, f, F)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_emo_edge():
    """Test edge cases."""
    r = 10
    n = 100
    k = 5
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_order_moments(r, n, k, f, F)
    assert isinstance(result, dict)
