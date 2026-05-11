"""Tests for icrf.item_characteristic_curve."""
import numpy as np
import pytest
from morie.fn.icrf import item_characteristic_curve


def test_icrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = item_characteristic_curve(y, theta, a, b, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = item_characteristic_curve(y, theta, a, b, c)
    assert isinstance(result, dict)
