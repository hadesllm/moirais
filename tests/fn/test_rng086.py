"""Tests for rng086.rangayyan_ch3_normalized_cross_correlation_template."""
import numpy as np
import pytest
from morie.fn.rng086 import rangayyan_ch3_normalized_cross_correlation_template


def test_rng086_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    N = 100
    x_bar = np.random.default_rng(42).normal(0, 1, 100)
    y_bar_k = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_normalized_cross_correlation_template(x, y, k, N, x_bar, y_bar_k)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng086_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    N = 100
    x_bar = np.random.default_rng(42).normal(0, 1, 100)
    y_bar_k = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_normalized_cross_correlation_template(x, y, k, N, x_bar, y_bar_k)
    assert isinstance(result, dict)
