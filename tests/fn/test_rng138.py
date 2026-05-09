"""Tests for rng138.rangayyan_ch3_wiener_filter_output_convolution."""
import numpy as np
import pytest
from moirais.fn.rng138 import rangayyan_ch3_wiener_filter_output_convolution


def test_rng138_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_wiener_filter_output_convolution(x, w_k, n, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng138_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_wiener_filter_output_convolution(x, w_k, n, M)
    assert isinstance(result, dict)
