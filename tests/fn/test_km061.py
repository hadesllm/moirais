"""Tests for km061.kamath_ch4_krona_output."""
import numpy as np
import pytest
from morie.fn.km061 import kamath_ch4_krona_output


def test_km061_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    A_k = np.random.default_rng(42).normal(0, 1, 100)
    B_k = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = kamath_ch4_krona_output(X, W, A_k, B_k, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km061_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    A_k = np.random.default_rng(42).normal(0, 1, 100)
    B_k = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = kamath_ch4_krona_output(X, W, A_k, B_k, s)
    assert isinstance(result, dict)
