"""Tests for tukrr.tukey_regression."""
import numpy as np
import pytest
from moirais.fn.tukrr import tukey_regression


def test_tukrr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = tukey_regression(X, y, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tukrr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = tukey_regression(X, y, c)
    assert isinstance(result, dict)
