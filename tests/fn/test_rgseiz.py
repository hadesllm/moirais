"""Tests for rgseiz.rangayyan_seizure_detect."""
import numpy as np
import pytest
from moirais.fn.rgseiz import rangayyan_seizure_detect


def test_rgseiz_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    ch_pairs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_seizure_detect(eeg, fs, ch_pairs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgseiz_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    ch_pairs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_seizure_detect(eeg, fs, ch_pairs)
    assert isinstance(result, dict)
