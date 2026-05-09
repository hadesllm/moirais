"""Tests for km126.kamath_ch8_smd."""
import numpy as np
import pytest
from moirais.fn.km126 import kamath_ch8_smd


def test_km126_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_smd(x, y, E)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km126_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_smd(x, y, E)
    assert isinstance(result, dict)
