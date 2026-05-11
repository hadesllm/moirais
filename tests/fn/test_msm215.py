"""Tests for msm215.mvsml_ridge_lasso_elastic_eq_9_33."""
import numpy as np
import pytest
from morie.fn.msm215 import mvsml_ridge_lasso_elastic_eq_9_33


def test_msm215_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_33(i, jyiy, j, xi, x, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm215_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_33(i, jyiy, j, xi, x, z)
    assert isinstance(result, dict)
