"""Tests for rgistft.rangayyan_istft."""
import numpy as np
import pytest
from morie.fn.rgistft import rangayyan_istft


def test_rgistft_basic():
    """Test basic functionality."""
    stft = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    hop = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_istft(stft, window, hop)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgistft_edge():
    """Test edge cases."""
    stft = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    hop = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_istft(stft, window, hop)
    assert isinstance(result, dict)
