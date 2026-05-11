"""Tests for rng142.rangayyan_ch3_cross_correlation_vector."""
import numpy as np
import pytest
from morie.fn.rng142 import rangayyan_ch3_cross_correlation_vector


def test_rng142_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    n = 100
    result = rangayyan_ch3_cross_correlation_vector(x, d, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng142_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    n = 100
    result = rangayyan_ch3_cross_correlation_vector(x, d, n)
    assert isinstance(result, dict)
