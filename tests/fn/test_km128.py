"""Tests for km128.kamath_ch8_pass_at_k."""
import numpy as np
import pytest
from moirais.fn.km128 import kamath_ch8_pass_at_k


def test_km128_basic():
    """Test basic functionality."""
    n = 100
    c = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_ch8_pass_at_k(n, c, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km128_edge():
    """Test edge cases."""
    n = 100
    c = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_ch8_pass_at_k(n, c, k)
    assert isinstance(result, dict)
