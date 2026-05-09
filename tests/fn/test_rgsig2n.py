"""Tests for rgsig2n.rangayyan_signal_to_noise."""
import numpy as np
import pytest
from moirais.fn.rgsig2n import rangayyan_signal_to_noise


def test_rgsig2n_basic():
    """Test basic functionality."""
    signal_clean = np.random.default_rng(42).normal(0, 1, 100)
    signal_noisy = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_signal_to_noise(signal_clean, signal_noisy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsig2n_edge():
    """Test edge cases."""
    signal_clean = np.random.default_rng(42).normal(0, 1, 100)
    signal_noisy = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_signal_to_noise(signal_clean, signal_noisy)
    assert isinstance(result, dict)
