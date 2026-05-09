"""Tests for rgemgfd.rangayyan_emg_fractal_dim."""
import numpy as np
import pytest
from moirais.fn.rgemgfd import rangayyan_emg_fractal_dim


def test_rgemgfd_basic():
    """Test basic functionality."""
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    force = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    kmax = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_emg_fractal_dim(emg, force, fs, kmax)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgemgfd_edge():
    """Test edge cases."""
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    force = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    kmax = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_emg_fractal_dim(emg, force, fs, kmax)
    assert isinstance(result, dict)
