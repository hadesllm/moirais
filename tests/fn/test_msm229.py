"""Tests for msm229.mvsml_ridge_lasso_elastic_eq_9_39."""
import numpy as np
import pytest
from moirais.fn.msm229 import mvsml_ridge_lasso_elastic_eq_9_39


def test_msm229_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    By = np.random.default_rng(42).normal(0, 1, 100)
    placing = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_39(i, n, yi, xT, By, placing)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm229_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    By = np.random.default_rng(42).normal(0, 1, 100)
    placing = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_39(i, n, yi, xT, By, placing)
    assert isinstance(result, dict)
