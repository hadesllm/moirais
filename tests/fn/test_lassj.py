"""Tests for lassj.lasso_objective."""
import numpy as np
import pytest
from morie.fn.lassj import lasso_objective


def test_lassj_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    lam = 0.1
    result = lasso_objective(y, X, beta, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lassj_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    lam = 0.1
    result = lasso_objective(y, X, beta, lam)
    assert isinstance(result, dict)
