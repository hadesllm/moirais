"""Tests for rgemgf.rangayyan_emg_force."""
import numpy as np
import pytest
from moirais.fn.rgemgf import rangayyan_emg_force


def test_rgemgf_basic():
    """Test basic functionality."""
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    force = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_emg_force(emg, force, fs, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgemgf_edge():
    """Test edge cases."""
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    force = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_emg_force(emg, force, fs, window)
    assert isinstance(result, dict)
