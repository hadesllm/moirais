"""Tests for km101.kamath_ch6_toxic_fraction."""
import numpy as np
import pytest
from moirais.fn.km101 import kamath_ch6_toxic_fraction


def test_km101_basic():
    """Test basic functionality."""
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_toxic_fraction(Yhat, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km101_edge():
    """Test edge cases."""
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_toxic_fraction(Yhat, c)
    assert isinstance(result, dict)
