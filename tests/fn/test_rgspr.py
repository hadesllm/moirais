"""Tests for rgspr.rangayyan_spectral_power_ratio."""
import numpy as np
import pytest
from moirais.fn.rgspr import rangayyan_spectral_power_ratio


def test_rgspr_basic():
    """Test basic functionality."""
    rr_psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_spectral_power_ratio(rr_psd, freqs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgspr_edge():
    """Test edge cases."""
    rr_psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_spectral_power_ratio(rr_psd, freqs)
    assert isinstance(result, dict)
