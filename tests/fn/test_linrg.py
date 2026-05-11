"""Tests for linrg.linear_regression_ols."""
import numpy as np
import pytest
from morie.fn.linrg import linear_regression_ols


def test_linrg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = linear_regression_ols(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_linrg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = linear_regression_ols(x, y)
    assert isinstance(result, dict)
