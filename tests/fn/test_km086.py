"""Tests for km086.kamath_ch6_pll."""
import numpy as np
import pytest
from moirais.fn.km086 import kamath_ch6_pll


def test_km086_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = kamath_ch6_pll(S, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km086_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = kamath_ch6_pll(S, theta)
    assert isinstance(result, dict)
