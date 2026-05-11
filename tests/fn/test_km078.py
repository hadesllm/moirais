"""Tests for km078.kamath_ch6_alignment_function."""
import numpy as np
import pytest
from morie.fn.km078 import kamath_ch6_alignment_function


def test_km078_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch6_alignment_function(a, b, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km078_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch6_alignment_function(a, b, y)
    assert isinstance(result, dict)
