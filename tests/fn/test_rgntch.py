"""Tests for rgntch.rangayyan_notch_filter."""
import numpy as np
import pytest
from morie.fn.rgntch import rangayyan_notch_filter


def test_rgntch_basic():
    """Test basic functionality."""
    notch_freq = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    fs = 100.0
    result = rangayyan_notch_filter(notch_freq, bandwidth, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgntch_edge():
    """Test edge cases."""
    notch_freq = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    fs = 100.0
    result = rangayyan_notch_filter(notch_freq, bandwidth, fs)
    assert isinstance(result, dict)
