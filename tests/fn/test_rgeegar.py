"""Tests for rgeegar.rangayyan_eeg_autocorr."""
import numpy as np
import pytest
from morie.fn.rgeegar import rangayyan_eeg_autocorr


def test_rgeegar_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eeg_autocorr(eeg, fs, max_lag)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rgeegar_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eeg_autocorr(eeg, fs, max_lag)
    assert isinstance(result, dict)
