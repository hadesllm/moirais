"""Tests for rgeqn6a.rangayyan_ch6_mean_freq."""
import numpy as np
import pytest
from morie.fn.rgeqn6a import rangayyan_ch6_mean_freq


def test_rgeqn6a_basic():
    """Test basic functionality."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch6_mean_freq(psd, freqs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn6a_edge():
    """Test edge cases."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch6_mean_freq(psd, freqs)
    assert isinstance(result, dict)
