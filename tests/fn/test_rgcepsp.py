"""Tests for rgcepsp.rangayyan_cepstrum_pitch."""
import numpy as np
import pytest
from morie.fn.rgcepsp import rangayyan_cepstrum_pitch


def test_rgcepsp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    f0_range = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_cepstrum_pitch(x, fs, f0_range)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcepsp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    f0_range = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_cepstrum_pitch(x, fs, f0_range)
    assert isinstance(result, dict)
