"""Tests for wsmbgn.wasserman_bagging."""
import numpy as np
import pytest
from moirais.fn.wsmbgn import wasserman_bagging


def test_wsmbgn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = wasserman_bagging(X, y, model, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmbgn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = wasserman_bagging(X, y, model, B)
    assert isinstance(result, dict)
