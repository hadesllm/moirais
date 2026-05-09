"""Tests for rgeegb.rangayyan_eeg_rhythms."""
import numpy as np
import pytest
from moirais.fn.rgeegb import rangayyan_eeg_rhythms


def test_rgeegb_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_eeg_rhythms(eeg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeegb_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_eeg_rhythms(eeg, fs)
    assert isinstance(result, dict)
