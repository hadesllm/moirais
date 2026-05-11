"""Tests for rgsmom.rangayyan_spectral_moments."""
import numpy as np
import pytest
from morie.fn.rgsmom import rangayyan_spectral_moments


def test_rgsmom_basic():
    """Test basic functionality."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_spectral_moments(psd, freqs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsmom_edge():
    """Test edge cases."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_spectral_moments(psd, freqs)
    assert isinstance(result, dict)
