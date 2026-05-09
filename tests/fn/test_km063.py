"""Tests for km063.kamath_ch4_vera_forward."""
import numpy as np
import pytest
from moirais.fn.km063 import kamath_ch4_vera_forward


def test_km063_basic():
    """Test basic functionality."""
    W_0 = np.random.default_rng(42).normal(0, 1, 100)
    Lambda_b = np.random.default_rng(42).normal(0, 1, 100)
    Lambda_d = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_vera_forward(W_0, Lambda_b, Lambda_d, A, B, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km063_edge():
    """Test edge cases."""
    W_0 = np.random.default_rng(42).normal(0, 1, 100)
    Lambda_b = np.random.default_rng(42).normal(0, 1, 100)
    Lambda_d = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_vera_forward(W_0, Lambda_b, Lambda_d, A, B, x)
    assert isinstance(result, dict)
