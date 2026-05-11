"""Tests for rgbbnorm.rangayyan_ecg_bbb_normal."""
import numpy as np
import pytest
from morie.fn.rgbbnorm import rangayyan_ecg_bbb_normal


def test_rgbbnorm_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ecg_bbb_normal(ecg, fs, r_peaks, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbbnorm_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ecg_bbb_normal(ecg, fs, r_peaks, labels)
    assert isinstance(result, dict)
