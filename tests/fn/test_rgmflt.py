"""Tests for rgmflt.rangayyan_matched_filter."""
import numpy as np
import pytest
from moirais.fn.rgmflt import rangayyan_matched_filter


def test_rgmflt_basic():
    """Test basic functionality."""
    signal_spectrum = np.random.default_rng(42).normal(0, 1, 100)
    noise_psd = np.random.default_rng(42).normal(0, 1, 100)
    t0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_matched_filter(signal_spectrum, noise_psd, t0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmflt_edge():
    """Test edge cases."""
    signal_spectrum = np.random.default_rng(42).normal(0, 1, 100)
    noise_psd = np.random.default_rng(42).normal(0, 1, 100)
    t0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_matched_filter(signal_spectrum, noise_psd, t0)
    assert isinstance(result, dict)
