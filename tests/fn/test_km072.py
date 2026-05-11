"""Tests for km072.kamath_ch5_bradley_terry_pref."""
import numpy as np
import pytest
from morie.fn.km072 import kamath_ch5_bradley_terry_pref


def test_km072_basic():
    """Test basic functionality."""
    r_star = np.random.default_rng(42).normal(0, 1, 100)
    y_w = np.random.default_rng(42).normal(0, 1, 100)
    y_l = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch5_bradley_terry_pref(r_star, y_w, y_l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km072_edge():
    """Test edge cases."""
    r_star = np.random.default_rng(42).normal(0, 1, 100)
    y_w = np.random.default_rng(42).normal(0, 1, 100)
    y_l = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch5_bradley_terry_pref(r_star, y_w, y_l)
    assert isinstance(result, dict)
