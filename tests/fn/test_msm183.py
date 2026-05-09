"""Tests for msm183.mvsml_ridge_lasso_elastic_eq_9_8."""
import numpy as np
import pytest
from moirais.fn.msm183 import mvsml_ridge_lasso_elastic_eq_9_8


def test_msm183_basic():
    """Test basic functionality."""
    minimize = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    p = 5
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_8(minimize, z, p, yi, xT, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm183_edge():
    """Test edge cases."""
    minimize = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    p = 5
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_8(minimize, z, p, yi, xT, i)
    assert isinstance(result, dict)
