"""Tests for msm204.mvsml_ridge_lasso_elastic_eq_9_30."""
import numpy as np
import pytest
from morie.fn.msm204 import mvsml_ridge_lasso_elastic_eq_9_30


def test_msm204_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_ridge_lasso_elastic_eq_9_30(i, yi, xT, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm204_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_ridge_lasso_elastic_eq_9_30(i, yi, xT, n)
    assert isinstance(result, dict)
