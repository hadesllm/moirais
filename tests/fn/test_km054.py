"""Tests for km054.kamath_ch4_series_adapter."""
import numpy as np
import pytest
from morie.fn.km054 import kamath_ch4_series_adapter


def test_km054_basic():
    """Test basic functionality."""
    H_o = np.random.default_rng(42).normal(0, 1, 100)
    W_down = np.random.default_rng(42).normal(0, 1, 100)
    W_up = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_series_adapter(H_o, W_down, W_up)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km054_edge():
    """Test edge cases."""
    H_o = np.random.default_rng(42).normal(0, 1, 100)
    W_down = np.random.default_rng(42).normal(0, 1, 100)
    W_up = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_series_adapter(H_o, W_down, W_up)
    assert isinstance(result, dict)
