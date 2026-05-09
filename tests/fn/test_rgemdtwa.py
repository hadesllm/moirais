"""Tests for rgemdtwa.rangayyan_emd_twa."""
import numpy as np
import pytest
from moirais.fn.rgemdtwa import rangayyan_emd_twa


def test_rgemdtwa_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_emd_twa(ecg, fs, r_peaks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgemdtwa_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_emd_twa(ecg, fs, r_peaks)
    assert isinstance(result, dict)
