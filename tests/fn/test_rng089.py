"""Tests for rng089.rangayyan_ch3_hann_filter."""
import numpy as np
import pytest
from morie.fn.rng089 import rangayyan_ch3_hann_filter


def test_rng089_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_hann_filter(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng089_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_hann_filter(x, n)
    assert isinstance(result, dict)
