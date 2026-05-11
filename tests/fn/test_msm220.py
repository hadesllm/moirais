"""Tests for msm220.mvsml_ridge_lasso_elastic_eq_9_36."""
import numpy as np
import pytest
from morie.fn.msm220 import mvsml_ridge_lasso_elastic_eq_9_36


def test_msm220_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    yi = np.random.default_rng(42).normal(0, 1, 100)
    j1xij = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_36(X, p, yi, j1xij, M, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm220_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    yi = np.random.default_rng(42).normal(0, 1, 100)
    j1xij = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_36(X, p, yi, j1xij, M, i)
    assert isinstance(result, dict)
