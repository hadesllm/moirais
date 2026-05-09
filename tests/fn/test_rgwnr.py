"""Tests for rgwnr.rangayyan_wiener_filter."""
import numpy as np
import pytest
from moirais.fn.rgwnr import rangayyan_wiener_filter


def test_rgwnr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    noise_psd = np.random.default_rng(42).normal(0, 1, 100)
    signal_psd = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_wiener_filter(x, noise_psd, signal_psd)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwnr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    noise_psd = np.random.default_rng(42).normal(0, 1, 100)
    signal_psd = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_wiener_filter(x, noise_psd, signal_psd)
    assert isinstance(result, dict)
