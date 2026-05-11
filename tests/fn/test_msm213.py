"""Tests for msm213.mvsml_ridge_lasso_elastic_eq_9_33."""
import numpy as np
import pytest
from morie.fn.msm213 import mvsml_ridge_lasso_elastic_eq_9_33


def test_msm213_basic():
    """Test basic functionality."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_33(L, i, jyiy, j, xi, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm213_edge():
    """Test edge cases."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_33(L, i, jyiy, j, xi, x)
    assert isinstance(result, dict)
