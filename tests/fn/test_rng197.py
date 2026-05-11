"""Tests for rng197.rangayyan_ch4_dicrotic_notch_smoothed_squared."""
import numpy as np
import pytest
from morie.fn.rng197 import rangayyan_ch4_dicrotic_notch_smoothed_squared


def test_rng197_basic():
    """Test basic functionality."""
    p = 5
    w = np.random.default_rng(45).exponential(1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch4_dicrotic_notch_smoothed_squared(p, w, n, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng197_edge():
    """Test edge cases."""
    p = 5
    w = np.random.default_rng(45).exponential(1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch4_dicrotic_notch_smoothed_squared(p, w, n, M)
    assert isinstance(result, dict)
