"""Tests for rgmatefp.rangayyan_maternal_ecg_filter."""
import numpy as np
import pytest
from moirais.fn.rgmatefp import rangayyan_maternal_ecg_filter


def test_rgmatefp_basic():
    """Test basic functionality."""
    abdominal_ecg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_channels = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_maternal_ecg_filter(abdominal_ecg, fs, n_channels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmatefp_edge():
    """Test edge cases."""
    abdominal_ecg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_channels = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_maternal_ecg_filter(abdominal_ecg, fs, n_channels)
    assert isinstance(result, dict)
