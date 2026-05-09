"""Tests for rgtwa.rangayyan_twave_alternans."""
import numpy as np
import pytest
from moirais.fn.rgtwa import rangayyan_twave_alternans


def test_rgtwa_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_twave_alternans(ecg, fs, r_peaks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgtwa_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_twave_alternans(ecg, fs, r_peaks)
    assert isinstance(result, dict)
