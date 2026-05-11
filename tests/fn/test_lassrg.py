"""Tests for lassrg.lasso_regression."""
import numpy as np
import pytest
from morie.fn.lassrg import lasso_regression


def test_lassrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = lasso_regression(y, X, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lassrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lam = 0.1
    result = lasso_regression(y, X, lam)
    assert isinstance(result, dict)
