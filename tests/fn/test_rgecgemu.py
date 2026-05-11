"""Tests for rgecgemu.rangayyan_ecg_emg_coupling."""
import numpy as np
import pytest
from morie.fn.rgecgemu import rangayyan_ecg_emg_coupling


def test_rgecgemu_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_ecg_emg_coupling(ecg, emg, fs)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rgecgemu_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_ecg_emg_coupling(ecg, emg, fs)
    assert isinstance(result, dict)
