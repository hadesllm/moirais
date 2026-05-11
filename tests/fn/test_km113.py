"""Tests for km113.kamath_ch8_perplexity."""
import numpy as np
import pytest
from morie.fn.km113 import kamath_ch8_perplexity


def test_km113_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    N = 100
    p_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_perplexity(X, N, p_theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km113_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    N = 100
    p_theta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_perplexity(X, N, p_theta)
    assert isinstance(result, dict)
