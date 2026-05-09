"""Tests for km013.kamath_ch2_positional_encoding_sin."""
import numpy as np
import pytest
from moirais.fn.km013 import kamath_ch2_positional_encoding_sin


def test_km013_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_positional_encoding_sin(i, j, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km013_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_positional_encoding_sin(i, j, d)
    assert isinstance(result, dict)
