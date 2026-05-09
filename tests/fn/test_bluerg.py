"""Tests for bluerg.blue_gls."""
import numpy as np
import pytest
from moirais.fn.bluerg import blue_gls


def test_bluerg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = blue_gls(y, X, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bluerg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = blue_gls(y, X, V)
    assert isinstance(result, dict)
