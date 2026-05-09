"""Tests for rgeegadp.rangayyan_eeg_adaptive_seg."""
import numpy as np
import pytest
from moirais.fn.rgeegadp import rangayyan_eeg_adaptive_seg


def test_rgeegadp_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    min_seg = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eeg_adaptive_seg(eeg, fs, min_seg, threshold)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rgeegadp_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    min_seg = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eeg_adaptive_seg(eeg, fs, min_seg, threshold)
    assert isinstance(result, dict)
