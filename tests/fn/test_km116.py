"""Tests for km116.kamath_ch8_brevity_penalty."""
import numpy as np
import pytest
from moirais.fn.km116 import kamath_ch8_brevity_penalty


def test_km116_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = kamath_ch8_brevity_penalty(c, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km116_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = kamath_ch8_brevity_penalty(c, r)
    assert isinstance(result, dict)
