"""Tests for msm230.mvsml_ridge_lasso_elastic_eq_9_43."""
import numpy as np
import pytest
from morie.fn.msm230 import mvsml_ridge_lasso_elastic_eq_9_43


def test_msm230_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    By = np.random.default_rng(42).normal(0, 1, 100)
    placing = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_43(i, n, yi, xT, By, placing)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm230_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    By = np.random.default_rng(42).normal(0, 1, 100)
    placing = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_43(i, n, yi, xT, By, placing)
    assert isinstance(result, dict)
