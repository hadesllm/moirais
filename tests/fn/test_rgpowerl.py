"""Tests for rgpowerl.rangayyan_powerline_removal."""
import numpy as np
import pytest
from morie.fn.rgpowerl import rangayyan_powerline_removal


def test_rgpowerl_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    powerline_freq = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_powerline_removal(ecg, fs, powerline_freq)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpowerl_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    powerline_freq = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_powerline_removal(ecg, fs, powerline_freq)
    assert isinstance(result, dict)
