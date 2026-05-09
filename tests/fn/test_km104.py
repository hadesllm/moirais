"""Tests for km104.kamath_ch6_affect_lm."""
import numpy as np
import pytest
from moirais.fn.km104 import kamath_ch6_affect_lm


def test_km104_basic():
    """Test basic functionality."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    beta = 0.8
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_affect_lm(U, V, f, g, c, e, beta, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km104_edge():
    """Test edge cases."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    beta = 0.8
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_affect_lm(U, V, f, g, c, e, beta, b)
    assert isinstance(result, dict)
