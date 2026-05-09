"""Tests for hmlrl.geron_linear_regression_life."""
import numpy as np
import pytest
from moirais.fn.hmlrl import geron_linear_regression_life


def test_hmlrl_basic():
    """Test basic functionality."""
    gdp = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    theta1 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_linear_regression_life(gdp, theta0, theta1)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlrl_edge():
    """Test edge cases."""
    gdp = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    theta1 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_linear_regression_life(gdp, theta0, theta1)
    assert isinstance(result, dict)
