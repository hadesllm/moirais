"""Tests for rng180.rangayyan_ch4_qrs_smoothing_ma_filter."""
import numpy as np
import pytest
from moirais.fn.rng180 import rangayyan_ch4_qrs_smoothing_ma_filter


def test_rng180_basic():
    """Test basic functionality."""
    g_1 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch4_qrs_smoothing_ma_filter(g_1, n, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng180_edge():
    """Test edge cases."""
    g_1 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch4_qrs_smoothing_ma_filter(g_1, n, M)
    assert isinstance(result, dict)
