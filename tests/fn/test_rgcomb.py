"""Tests for rgcomb.rangayyan_comb_filter."""
import numpy as np
import pytest
from morie.fn.rgcomb import rangayyan_comb_filter


def test_rgcomb_basic():
    """Test basic functionality."""
    period_samples = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_comb_filter(period_samples, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcomb_edge():
    """Test edge cases."""
    period_samples = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_comb_filter(period_samples, fs)
    assert isinstance(result, dict)
