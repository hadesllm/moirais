"""Tests for rghrvf.rangayyan_hrv_freq_domain."""
import numpy as np
import pytest
from morie.fn.rghrvf import rangayyan_hrv_freq_domain


def test_rghrvf_basic():
    """Test basic functionality."""
    rr_intervals = np.random.default_rng(42).normal(0, 1, 100)
    fs_resamp = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hrv_freq_domain(rr_intervals, fs_resamp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghrvf_edge():
    """Test edge cases."""
    rr_intervals = np.random.default_rng(42).normal(0, 1, 100)
    fs_resamp = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hrv_freq_domain(rr_intervals, fs_resamp)
    assert isinstance(result, dict)
