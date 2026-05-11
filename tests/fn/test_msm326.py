"""Tests for msm326.mvsml_functional_regression_eq_15_2."""
import numpy as np
import pytest
from morie.fn.msm326 import mvsml_functional_regression_eq_15_2


def test_msm326_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    exp = np.random.default_rng(42).normal(0, 1, 100)
    For = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_2(Y, i, N, exp, For, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm326_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    exp = np.random.default_rng(42).normal(0, 1, 100)
    For = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_2(Y, i, N, exp, For, a)
    assert isinstance(result, dict)
