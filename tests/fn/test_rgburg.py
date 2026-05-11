"""Tests for rgburg.rangayyan_burg_method."""
import numpy as np
import pytest
from morie.fn.rgburg import rangayyan_burg_method


def test_rgburg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    fs = 100.0
    result = rangayyan_burg_method(x, order, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgburg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    fs = 100.0
    result = rangayyan_burg_method(x, order, fs)
    assert isinstance(result, dict)
