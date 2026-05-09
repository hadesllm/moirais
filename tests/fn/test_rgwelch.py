"""Tests for rgwelch.rangayyan_welch_psd."""
import numpy as np
import pytest
from moirais.fn.rgwelch import rangayyan_welch_psd


def test_rgwelch_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    nperseg = np.random.default_rng(42).normal(0, 1, 100)
    noverlap = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_welch_psd(x, fs, nperseg, noverlap, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwelch_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    nperseg = np.random.default_rng(42).normal(0, 1, 100)
    noverlap = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_welch_psd(x, fs, nperseg, noverlap, window)
    assert isinstance(result, dict)
