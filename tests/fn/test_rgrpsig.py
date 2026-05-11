"""Tests for rgrpsig.rangayyan_resp_signal."""
import numpy as np
import pytest
from morie.fn.rgrpsig import rangayyan_resp_signal


def test_rgrpsig_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    r_peaks = np.arange(50, 1000, 50)
    fs_out = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_resp_signal(ecg, r_peaks, fs_out)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgrpsig_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    r_peaks = np.arange(50, 1000, 50)
    fs_out = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_resp_signal(ecg, r_peaks, fs_out)
    assert isinstance(result, dict)
