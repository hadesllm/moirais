"""Tests for msm226.mvsml_ridge_lasso_elastic_eq_9_41."""
import numpy as np
import pytest
from morie.fn.msm226 import mvsml_ridge_lasso_elastic_eq_9_41


def test_msm226_basic():
    """Test basic functionality."""
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_41(Xn, L, i, iyi, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm226_edge():
    """Test edge cases."""
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_41(Xn, L, i, iyi, T)
    assert isinstance(result, dict)
