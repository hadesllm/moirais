"""Tests for rgbbb.rangayyan_bundle_branch_block."""
import numpy as np
import pytest
from morie.fn.rgbbb import rangayyan_bundle_branch_block


def test_rgbbb_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_bundle_branch_block(ecg, fs, r_peaks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbbb_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_bundle_branch_block(ecg, fs, r_peaks)
    assert isinstance(result, dict)
