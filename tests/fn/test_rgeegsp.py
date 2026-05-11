"""Tests for rgeegsp.rangayyan_eeg_spectral."""
import numpy as np
import pytest
from morie.fn.rgeegsp import rangayyan_eeg_spectral


def test_rgeegsp_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    n_ch = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eeg_spectral(eeg, fs, n_ch)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeegsp_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    n_ch = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eeg_spectral(eeg, fs, n_ch)
    assert isinstance(result, dict)
