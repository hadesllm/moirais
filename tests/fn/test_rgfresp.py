"""Tests for rgfresp.rangayyan_freq_response."""
import numpy as np
import pytest
from morie.fn.rgfresp import rangayyan_freq_response


def test_rgfresp_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    fs = 100.0
    n_freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_freq_response(b, a, fs, n_freqs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfresp_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    fs = 100.0
    n_freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_freq_response(b, a, fs, n_freqs)
    assert isinstance(result, dict)
