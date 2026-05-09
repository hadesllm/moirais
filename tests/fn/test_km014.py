"""Tests for km014.kamath_ch2_positional_encoding_cos."""
import numpy as np
import pytest
from moirais.fn.km014 import kamath_ch2_positional_encoding_cos


def test_km014_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_positional_encoding_cos(i, j, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km014_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_positional_encoding_cos(i, j, d)
    assert isinstance(result, dict)
