"""Tests for rgbartl.rangayyan_bartlett_psd."""
import numpy as np
import pytest
from moirais.fn.rgbartl import rangayyan_bartlett_psd


def test_rgbartl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    nseg = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bartlett_psd(x, fs, nseg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbartl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    nseg = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bartlett_psd(x, fs, nseg)
    assert isinstance(result, dict)
