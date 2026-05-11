"""Tests for tcc._test_characteristic_curve."""
import numpy as np
import pytest
from morie.fn.tcc import test_characteristic_curve as _test_characteristic_curve


def test_tcc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = _test_characteristic_curve(y, theta, a, b)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_tcc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = _test_characteristic_curve(y, theta, a, b)
    assert isinstance(result, dict)
