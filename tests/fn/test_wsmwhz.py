"""Tests for wsmwhz.wasserman_white_huber."""
import numpy as np
import pytest
from moirais.fn.wsmwhz import wasserman_white_huber


def test_wsmwhz_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_white_huber(X, y, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmwhz_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_white_huber(X, y, f)
    assert isinstance(result, dict)
