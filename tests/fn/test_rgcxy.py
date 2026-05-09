"""Tests for rgcxy.rangayyan_coherence_cxy."""
import numpy as np
import pytest
from moirais.fn.rgcxy import rangayyan_coherence_cxy


def test_rgcxy_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    fs = 100.0
    nperseg = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_coherence_cxy(x, y, fs, nperseg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcxy_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    fs = 100.0
    nperseg = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_coherence_cxy(x, y, fs, nperseg)
    assert isinstance(result, dict)
