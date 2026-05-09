"""Tests for msm223.mvsml_ridge_lasso_elastic_eq_9_38."""
import numpy as np
import pytest
from moirais.fn.msm223 import mvsml_ridge_lasso_elastic_eq_9_38


def test_msm223_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_38(i, yi, xT, L, e, Xn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm223_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_38(i, yi, xT, L, e, Xn)
    assert isinstance(result, dict)
