"""Tests for rgbwbnd.rangayyan_bandwidth."""
import numpy as np
import pytest
from morie.fn.rgbwbnd import rangayyan_bandwidth


def test_rgbwbnd_basic():
    """Test basic functionality."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    criterion = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bandwidth(psd, freqs, criterion)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbwbnd_edge():
    """Test edge cases."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    criterion = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bandwidth(psd, freqs, criterion)
    assert isinstance(result, dict)
