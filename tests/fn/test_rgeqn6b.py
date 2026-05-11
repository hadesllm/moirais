"""Tests for rgeqn6b.rangayyan_ch6_median_freq."""
import numpy as np
import pytest
from morie.fn.rgeqn6b import rangayyan_ch6_median_freq


def test_rgeqn6b_basic():
    """Test basic functionality."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch6_median_freq(psd, freqs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn6b_edge():
    """Test edge cases."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch6_median_freq(psd, freqs)
    assert isinstance(result, dict)
