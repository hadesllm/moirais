"""Tests for km016.kamath_ch2_multihead_concat."""
import numpy as np
import pytest
from moirais.fn.km016 import kamath_ch2_multihead_concat


def test_km016_basic():
    """Test basic functionality."""
    heads = np.random.default_rng(42).normal(0, 1, 100)
    W_O = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_multihead_concat(heads, W_O)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km016_edge():
    """Test edge cases."""
    heads = np.random.default_rng(42).normal(0, 1, 100)
    W_O = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_multihead_concat(heads, W_O)
    assert isinstance(result, dict)
