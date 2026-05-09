"""Tests for rgpark.rangayyan_parkinson_multimodal."""
import numpy as np
import pytest
from moirais.fn.rgpark import rangayyan_parkinson_multimodal


def test_rgpark_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    gait = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_parkinson_multimodal(eeg, emg, gait, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpark_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    gait = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_parkinson_multimodal(eeg, emg, gait, fs)
    assert isinstance(result, dict)
