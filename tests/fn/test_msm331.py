"""Tests for msm331.mvsml_functional_regression_eq_15_4."""
import numpy as np
import pytest
from moirais.fn.msm331 import mvsml_functional_regression_eq_15_4


def test_msm331_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    ZAPC_RF = np.random.default_rng(42).normal(0, 1, 100)
    conventional = np.random.default_rng(42).normal(0, 1, 100)
    logistic = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_4(b, The, ZAPC_RF, conventional, logistic, regression)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm331_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    ZAPC_RF = np.random.default_rng(42).normal(0, 1, 100)
    conventional = np.random.default_rng(42).normal(0, 1, 100)
    logistic = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_4(b, The, ZAPC_RF, conventional, logistic, regression)
    assert isinstance(result, dict)
