"""Tests for msm205.mvsml_ridge_lasso_elastic_eq_9_30."""
import numpy as np
import pytest
from morie.fn.msm205 import mvsml_ridge_lasso_elastic_eq_9_30


def test_msm205_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    conditions = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_30(yi, xT, i, The, conditions, that)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm205_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    conditions = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_30(yi, xT, i, The, conditions, that)
    assert isinstance(result, dict)
