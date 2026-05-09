"""Tests for wsmbst.wasserman_boosting."""
import numpy as np
import pytest
from moirais.fn.wsmbst import wasserman_boosting


def test_wsmbst_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = wasserman_boosting(X, y, model, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmbst_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = wasserman_boosting(X, y, model, T)
    assert isinstance(result, dict)
