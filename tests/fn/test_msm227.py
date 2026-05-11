"""Tests for msm227.mvsml_ridge_lasso_elastic_eq_9_42."""
import numpy as np
import pytest
from morie.fn.msm227 import mvsml_ridge_lasso_elastic_eq_9_42


def test_msm227_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_ridge_lasso_elastic_eq_9_42(i, yi, xT, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm227_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_ridge_lasso_elastic_eq_9_42(i, yi, xT, n)
    assert isinstance(result, dict)
