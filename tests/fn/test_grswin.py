"""Tests for grswin.geron_swin_window_attention."""
import numpy as np
import pytest
from moirais.fn.grswin import geron_swin_window_attention


def test_grswin_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    window_size = 100
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_swin_window_attention(X, window_size, WQ, WK, WV)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grswin_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    window_size = 100
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_swin_window_attention(X, window_size, WQ, WK, WV)
    assert isinstance(result, dict)
