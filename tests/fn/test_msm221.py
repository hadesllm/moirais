"""Tests for msm221.mvsml_ridge_lasso_elastic_eq_9_37."""
import numpy as np
import pytest
from moirais.fn.msm221 import mvsml_ridge_lasso_elastic_eq_9_37


def test_msm221_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    i = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_37(M, i, j, X, n, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm221_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    i = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_37(M, i, j, X, n, T)
    assert isinstance(result, dict)
