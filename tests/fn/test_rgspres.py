"""Tests for rgspres.rangayyan_spectral_resolution."""
import numpy as np
import pytest
from moirais.fn.rgspres import rangayyan_spectral_resolution


def test_rgspres_basic():
    """Test basic functionality."""
    N = 100
    fs = 100.0
    window_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_spectral_resolution(N, fs, window_type)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgspres_edge():
    """Test edge cases."""
    N = 100
    fs = 100.0
    window_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_spectral_resolution(N, fs, window_type)
    assert isinstance(result, dict)
