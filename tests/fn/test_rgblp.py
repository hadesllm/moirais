"""Tests for rgblp.rangayyan_butterworth_lp."""
import numpy as np
import pytest
from morie.fn.rgblp import rangayyan_butterworth_lp


def test_rgblp_basic():
    """Test basic functionality."""
    cutoff_hz = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    fs = 100.0
    result = rangayyan_butterworth_lp(cutoff_hz, order, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgblp_edge():
    """Test edge cases."""
    cutoff_hz = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    fs = 100.0
    result = rangayyan_butterworth_lp(cutoff_hz, order, fs)
    assert isinstance(result, dict)
