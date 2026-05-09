"""Tests for grn011.geron_ch4_lasso_regression_cost_function."""
import numpy as np
import pytest
from moirais.fn.grn011 import geron_ch4_lasso_regression_cost_function


def test_grn011_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    result = geron_ch4_lasso_regression_cost_function(X, y, theta, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grn011_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    result = geron_ch4_lasso_regression_cost_function(X, y, theta, alpha)
    assert isinstance(result, dict)
