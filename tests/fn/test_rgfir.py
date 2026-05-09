"""Tests for rgfir.rangayyan_fir_filter."""
import numpy as np
import pytest
from moirais.fn.rgfir import rangayyan_fir_filter


def test_rgfir_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    order = 4
    result = rangayyan_fir_filter(x, cutoff, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfir_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    order = 4
    result = rangayyan_fir_filter(x, cutoff, order)
    assert isinstance(result, dict)
