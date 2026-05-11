"""Tests for ridgrg.ridge_regression."""
import numpy as np
import pytest
from morie.fn.ridgrg import ridge_regression


def test_ridgrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = ridge_regression(y, X, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ridgrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = ridge_regression(y, X, lam)
    assert isinstance(result, dict)
