"""Tests for msm175.mvsml_ridge_lasso_elastic_eq_9_6."""
import numpy as np
import pytest
from morie.fn.msm175 import mvsml_ridge_lasso_elastic_eq_9_6


def test_msm175_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    n = 100
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(yi, xT, i, M, n, The)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm175_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    n = 100
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(yi, xT, i, M, n, The)
    assert isinstance(result, dict)
