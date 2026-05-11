"""Tests for sarimax.sarimax."""
import numpy as np
import pytest
from morie.fn.sarimax import sarimax


def test_sarimax_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    d = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = sarimax(y, X, p, d, q, P, D, Q, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sarimax_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    d = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = sarimax(y, X, p, d, q, P, D, Q, s)
    assert isinstance(result, dict)
