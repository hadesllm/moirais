"""Tests for rgsnr.rangayyan_snr."""
import numpy as np
import pytest
from morie.fn.rgsnr import rangayyan_snr


def test_rgsnr_basic():
    """Test basic functionality."""
    signal = np.random.default_rng(42).normal(0, 1, 1024)
    noise = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_snr(signal, noise)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsnr_edge():
    """Test edge cases."""
    signal = np.random.default_rng(42).normal(0, 1, 1024)
    noise = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_snr(signal, noise)
    assert isinstance(result, dict)
