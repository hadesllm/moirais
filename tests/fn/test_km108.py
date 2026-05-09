"""Tests for km108.kamath_ch6_differential_privacy."""
import numpy as np
import pytest
from moirais.fn.km108 import kamath_ch6_differential_privacy


def test_km108_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    S = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = kamath_ch6_differential_privacy(M, A, B, S, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km108_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    S = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = kamath_ch6_differential_privacy(M, A, B, S, epsilon)
    assert isinstance(result, dict)
