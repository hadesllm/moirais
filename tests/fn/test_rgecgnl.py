"""Tests for rgecgnl.rangayyan_ecg_normal_ectopic."""
import numpy as np
import pytest
from morie.fn.rgecgnl import rangayyan_ecg_normal_ectopic


def test_rgecgnl_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_ecg_normal_ectopic(ecg, fs, r_peaks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgecgnl_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_ecg_normal_ectopic(ecg, fs, r_peaks)
    assert isinstance(result, dict)
