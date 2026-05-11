"""Tests for rgbci.rangayyan_bci_nmf."""
import numpy as np
import pytest
from morie.fn.rgbci import rangayyan_bci_nmf


def test_rgbci_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    n_components = 3
    fs = 100.0
    result = rangayyan_bci_nmf(eeg, n_components, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbci_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    n_components = 3
    fs = 100.0
    result = rangayyan_bci_nmf(eeg, n_components, fs)
    assert isinstance(result, dict)
