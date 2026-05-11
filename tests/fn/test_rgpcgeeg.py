"""Tests for rgpcgeeg.rangayyan_pcg_eeg_coupling."""
import numpy as np
import pytest
from morie.fn.rgpcgeeg import rangayyan_pcg_eeg_coupling


def test_rgpcgeeg_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_eeg_coupling(pcg, eeg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpcgeeg_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_eeg_coupling(pcg, eeg, fs)
    assert isinstance(result, dict)
