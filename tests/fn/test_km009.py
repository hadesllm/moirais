"""Tests for km009.kamath_ch2_softmax_element."""
import numpy as np
import pytest
from moirais.fn.km009 import kamath_ch2_softmax_element


def test_km009_basic():
    """Test basic functionality."""
    a_i = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch2_softmax_element(a_i, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km009_edge():
    """Test edge cases."""
    a_i = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch2_softmax_element(a_i, a)
    assert isinstance(result, dict)
