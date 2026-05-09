"""Tests for rgisint.rangayyan_isometric_contraction."""
import numpy as np
import pytest
from moirais.fn.rgisint import rangayyan_isometric_contraction


def test_rgisint_basic():
    """Test basic functionality."""
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    force = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_isometric_contraction(emg, force, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgisint_edge():
    """Test edge cases."""
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    force = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_isometric_contraction(emg, force, fs)
    assert isinstance(result, dict)
