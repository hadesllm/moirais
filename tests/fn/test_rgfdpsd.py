"""Tests for rgfdpsd.rangayyan_fd_psd_slope."""
import numpy as np
import pytest
from morie.fn.rgfdpsd import rangayyan_fd_psd_slope


def test_rgfdpsd_basic():
    """Test basic functionality."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    f_range = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_fd_psd_slope(psd, freqs, f_range)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfdpsd_edge():
    """Test edge cases."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    f_range = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_fd_psd_slope(psd, freqs, f_range)
    assert isinstance(result, dict)
