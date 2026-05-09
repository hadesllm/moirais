"""Tests for km053.kamath_ch3_prefix_tuning_obj."""
import numpy as np
import pytest
from moirais.fn.km053 import kamath_ch3_prefix_tuning_obj


def test_km053_basic():
    """Test basic functionality."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    result = kamath_ch3_prefix_tuning_obj(phi, x, y, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km053_edge():
    """Test edge cases."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    result = kamath_ch3_prefix_tuning_obj(phi, x, y, h)
    assert isinstance(result, dict)
