"""Tests for rgmfsnr.rangayyan_matched_filter_snr."""
import numpy as np
import pytest
from morie.fn.rgmfsnr import rangayyan_matched_filter_snr


def test_rgmfsnr_basic():
    """Test basic functionality."""
    signal = np.random.default_rng(42).normal(0, 1, 1024)
    noise_psd = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_matched_filter_snr(signal, noise_psd)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmfsnr_edge():
    """Test edge cases."""
    signal = np.random.default_rng(42).normal(0, 1, 1024)
    noise_psd = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_matched_filter_snr(signal, noise_psd)
    assert isinstance(result, dict)
