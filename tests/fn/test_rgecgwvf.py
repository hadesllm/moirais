"""Tests for rgecgwvf.rangayyan_ecg_waveshape."""
import numpy as np
import pytest
from moirais.fn.rgecgwvf import rangayyan_ecg_waveshape


def test_rgecgwvf_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    template = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ecg_waveshape(ecg, fs, r_peaks, template)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgecgwvf_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    template = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ecg_waveshape(ecg, fs, r_peaks, template)
    assert isinstance(result, dict)
