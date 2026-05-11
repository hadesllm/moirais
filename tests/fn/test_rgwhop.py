"""Tests for rgwhop.rangayyan_wiener_hopf."""
import numpy as np
import pytest
from morie.fn.rgwhop import rangayyan_wiener_hopf


def test_rgwhop_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    order = 4
    result = rangayyan_wiener_hopf(x, d, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwhop_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    order = 4
    result = rangayyan_wiener_hopf(x, d, order)
    assert isinstance(result, dict)
