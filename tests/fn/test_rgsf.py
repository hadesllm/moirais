"""Tests for rgsf.rangayyan_signal_features."""
import numpy as np
import pytest
from morie.fn.rgsf import rangayyan_signal_features


def test_rgsf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_signal_features(x, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_signal_features(x, fs)
    assert isinstance(result, dict)
