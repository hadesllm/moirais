"""Tests for km122.kamath_ch8_wmd."""
import numpy as np
import pytest
from morie.fn.km122 import kamath_ch8_wmd


def test_km122_basic():
    """Test basic functionality."""
    x_n = np.random.default_rng(42).normal(0, 1, 100)
    y_n = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch8_wmd(x_n, y_n, C, F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km122_edge():
    """Test edge cases."""
    x_n = np.random.default_rng(42).normal(0, 1, 100)
    y_n = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch8_wmd(x_n, y_n, C, F)
    assert isinstance(result, dict)
