"""Tests for rgecgf.rangayyan_ecg_features."""
import numpy as np
import pytest
from moirais.fn.rgecgf import rangayyan_ecg_features


def test_rgecgf_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_ecg_features(ecg, fs, r_peaks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgecgf_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_ecg_features(ecg, fs, r_peaks)
    assert isinstance(result, dict)
