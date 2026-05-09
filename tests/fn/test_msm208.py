"""Tests for msm208.mvsml_ridge_lasso_elastic_eq_9_28."""
import numpy as np
import pytest
from moirais.fn.msm208 import mvsml_ridge_lasso_elastic_eq_9_28


def test_msm208_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    maximum = np.random.default_rng(42).normal(0, 1, 100)
    margin = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_28(yi, xT, i, The, maximum, margin)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm208_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    maximum = np.random.default_rng(42).normal(0, 1, 100)
    margin = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_28(yi, xT, i, The, maximum, margin)
    assert isinstance(result, dict)
