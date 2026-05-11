"""Tests for km067.kamath_ch5_rm_bradley_terry."""
import numpy as np
import pytest
from morie.fn.km067 import kamath_ch5_rm_bradley_terry


def test_km067_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_w = np.random.default_rng(42).normal(0, 1, 100)
    y_l = np.random.default_rng(42).normal(0, 1, 100)
    r_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch5_rm_bradley_terry(x, y_w, y_l, r_theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km067_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_w = np.random.default_rng(42).normal(0, 1, 100)
    y_l = np.random.default_rng(42).normal(0, 1, 100)
    r_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch5_rm_bradley_terry(x, y_w, y_l, r_theta)
    assert isinstance(result, dict)
