"""Tests for km097.kamath_ch6_ear_entropy_reg."""
import numpy as np
import pytest
from moirais.fn.km097 import kamath_ch6_ear_entropy_reg


def test_km097_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    L = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_ch6_ear_entropy_reg(A, L, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km097_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    L = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_ch6_ear_entropy_reg(A, L, lam)
    assert isinstance(result, dict)
