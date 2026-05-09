"""Tests for msm201.mvsml_ridge_lasso_elastic_eq_9_27."""
import numpy as np
import pytest
from moirais.fn.msm201 import mvsml_ridge_lasso_elastic_eq_9_27


def test_msm201_basic():
    """Test basic functionality."""
    k = 5
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_27(k, k2, i, yi, xT, L)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm201_edge():
    """Test edge cases."""
    k = 5
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_27(k, k2, i, yi, xT, L)
    assert isinstance(result, dict)
