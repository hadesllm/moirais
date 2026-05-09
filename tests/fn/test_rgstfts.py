"""Tests for rgstfts.rangayyan_stft_spectrogram."""
import numpy as np
import pytest
from moirais.fn.rgstfts import rangayyan_stft_spectrogram


def test_rgstfts_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    nperseg = np.random.default_rng(42).normal(0, 1, 100)
    noverlap = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_stft_spectrogram(x, fs, nperseg, noverlap, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgstfts_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    nperseg = np.random.default_rng(42).normal(0, 1, 100)
    noverlap = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_stft_spectrogram(x, fs, nperseg, noverlap, window)
    assert isinstance(result, dict)
