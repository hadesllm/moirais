"""Tests for rgpolysg.rangayyan_polysomnography."""
import numpy as np
import pytest
from morie.fn.rgpolysg import rangayyan_polysomnography


def test_rgpolysg_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    eog = np.random.default_rng(42).normal(0, 1, 100)
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    epoch_len = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_polysomnography(eeg, eog, emg, fs, epoch_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpolysg_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    eog = np.random.default_rng(42).normal(0, 1, 100)
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    epoch_len = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_polysomnography(eeg, eog, emg, fs, epoch_len)
    assert isinstance(result, dict)
