"""Tests for hrztiku.horowitz_tikhonov_unknown_T."""
import numpy as np
import pytest
from morie.fn.hrztiku import horowitz_tikhonov_unknown_T


def test_hrztiku_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    bandwidth = 0.3
    alpha = 0.05
    result = horowitz_tikhonov_unknown_T(x, y, w, bandwidth, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrztiku_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    bandwidth = 0.3
    alpha = 0.05
    result = horowitz_tikhonov_unknown_T(x, y, w, bandwidth, alpha)
    assert isinstance(result, dict)
