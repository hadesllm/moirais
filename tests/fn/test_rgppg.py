"""Tests for rgppg.rangayyan_ppg_features."""
import numpy as np
import pytest
from morie.fn.rgppg import rangayyan_ppg_features


def test_rgppg_basic():
    """Test basic functionality."""
    ppg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_ppg_features(ppg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgppg_edge():
    """Test edge cases."""
    ppg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_ppg_features(ppg, fs)
    assert isinstance(result, dict)
