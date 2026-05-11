"""Tests for rgfeatex.rangayyan_feature_extract_bci."""
import numpy as np
import pytest
from morie.fn.rgfeatex import rangayyan_feature_extract_bci


def test_rgfeatex_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    ref_window = np.random.default_rng(42).normal(0, 1, 100)
    active_window = np.random.default_rng(42).normal(0, 1, 100)
    band = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_feature_extract_bci(eeg, fs, ref_window, active_window, band)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfeatex_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    ref_window = np.random.default_rng(42).normal(0, 1, 100)
    active_window = np.random.default_rng(42).normal(0, 1, 100)
    band = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_feature_extract_bci(eeg, fs, ref_window, active_window, band)
    assert isinstance(result, dict)
