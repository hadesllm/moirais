"""Tests for km123.kamath_ch8_moverscore_distance."""
import numpy as np
import pytest
from morie.fn.km123 import kamath_ch8_moverscore_distance


def test_km123_basic():
    """Test basic functionality."""
    x_i = np.random.default_rng(42).normal(0, 1, 100)
    y_j = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_moverscore_distance(x_i, y_j, E)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km123_edge():
    """Test edge cases."""
    x_i = np.random.default_rng(42).normal(0, 1, 100)
    y_j = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_moverscore_distance(x_i, y_j, E)
    assert isinstance(result, dict)
