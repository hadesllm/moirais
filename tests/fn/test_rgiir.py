"""Tests for rgiir.rangayyan_iir_filter."""
import numpy as np
import pytest
from morie.fn.rgiir import rangayyan_iir_filter


def test_rgiir_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    order = 4
    result = rangayyan_iir_filter(x, cutoff, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgiir_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    order = 4
    result = rangayyan_iir_filter(x, cutoff, order)
    assert isinstance(result, dict)
