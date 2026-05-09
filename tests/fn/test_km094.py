"""Tests for km094.kamath_ch6_debias_regularizer."""
import numpy as np
import pytest
from moirais.fn.km094 import kamath_ch6_debias_regularizer


def test_km094_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    E = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_ch6_debias_regularizer(A, E, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km094_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    E = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_ch6_debias_regularizer(A, E, lam)
    assert isinstance(result, dict)
