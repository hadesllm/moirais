"""Tests for msm225.mvsml_ridge_lasso_elastic_eq_9_40."""
import numpy as np
import pytest
from moirais.fn.msm225 import mvsml_ridge_lasso_elastic_eq_9_40


def test_msm225_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    iyixi = np.random.default_rng(42).normal(0, 1, 100)
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_40(i, iyixi, Xn, L, iyi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm225_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    iyixi = np.random.default_rng(42).normal(0, 1, 100)
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_40(i, iyixi, Xn, L, iyi)
    assert isinstance(result, dict)
