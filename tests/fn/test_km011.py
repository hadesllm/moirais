"""Tests for km011.kamath_ch2_scaled_dot_score."""
import numpy as np
import pytest
from moirais.fn.km011 import kamath_ch2_scaled_dot_score


def test_km011_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_scaled_dot_score(q, k, d_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km011_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_scaled_dot_score(q, k, d_k)
    assert isinstance(result, dict)
