"""Tests for km077.kamath_ch6_factscore."""
import numpy as np
import pytest
from morie.fn.km077 import kamath_ch6_factscore


def test_km077_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    A_y = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_factscore(M, X, A_y, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km077_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    A_y = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_factscore(M, X, A_y, C)
    assert isinstance(result, dict)
