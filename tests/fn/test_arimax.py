"""Tests for arimax.arimax."""
import numpy as np
import pytest
from moirais.fn.arimax import arimax


def test_arimax_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    d = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = arimax(y, X, p, d, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_arimax_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    d = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = arimax(y, X, p, d, q)
    assert isinstance(result, dict)
