"""Tests for rghrvtv.rangayyan_hrv_time_varying."""
import numpy as np
import pytest
from moirais.fn.rghrvtv import rangayyan_hrv_time_varying


def test_rghrvtv_basic():
    """Test basic functionality."""
    rr_intervals = np.random.default_rng(42).normal(0, 1, 100)
    fs_resamp = np.random.default_rng(42).normal(0, 1, 100)
    window_len = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hrv_time_varying(rr_intervals, fs_resamp, window_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghrvtv_edge():
    """Test edge cases."""
    rr_intervals = np.random.default_rng(42).normal(0, 1, 100)
    fs_resamp = np.random.default_rng(42).normal(0, 1, 100)
    window_len = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hrv_time_varying(rr_intervals, fs_resamp, window_len)
    assert isinstance(result, dict)
