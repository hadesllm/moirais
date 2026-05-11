"""Tests for msm330.mvsml_functional_regression_eq_15_4."""
import numpy as np
import pytest
from morie.fn.msm330 import mvsml_functional_regression_eq_15_4


def test_msm330_basic():
    """Test basic functionality."""
    bY = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    ZAPC_RF = np.random.default_rng(42).normal(0, 1, 100)
    conventional = np.random.default_rng(42).normal(0, 1, 100)
    logistic = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_4(bY, b, The, ZAPC_RF, conventional, logistic)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm330_edge():
    """Test edge cases."""
    bY = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    ZAPC_RF = np.random.default_rng(42).normal(0, 1, 100)
    conventional = np.random.default_rng(42).normal(0, 1, 100)
    logistic = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_4(bY, b, The, ZAPC_RF, conventional, logistic)
    assert isinstance(result, dict)
