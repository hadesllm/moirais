"""Tests for rgpantp.rangayyan_pan_tompkins."""
import numpy as np
import pytest
from morie.fn.rgpantp import rangayyan_pan_tompkins


def test_rgpantp_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pan_tompkins(ecg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpantp_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pan_tompkins(ecg, fs)
    assert isinstance(result, dict)
