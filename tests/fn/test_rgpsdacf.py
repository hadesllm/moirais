"""Tests for rgpsdacf.rangayyan_psd_to_acf."""
import numpy as np
import pytest
from morie.fn.rgpsdacf import rangayyan_psd_to_acf


def test_rgpsdacf_basic():
    """Test basic functionality."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_psd_to_acf(psd, freqs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpsdacf_edge():
    """Test edge cases."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_psd_to_acf(psd, freqs)
    assert isinstance(result, dict)
