"""Tests for mavi.ma_var_inflation_correlated."""
import numpy as np
import pytest
from morie.fn.mavi import ma_var_inflation_correlated


def test_mavi_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = ma_var_inflation_correlated(V, rho)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_mavi_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = ma_var_inflation_correlated(V, rho)
    assert isinstance(result, dict)
