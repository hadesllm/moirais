"""Tests for msm214.mvsml_ridge_lasso_elastic_eq_9_32."""
import numpy as np
import pytest
from moirais.fn.msm214 import mvsml_ridge_lasso_elastic_eq_9_32


def test_msm214_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_32(i, jyiy, j, xi, x, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm214_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_32(i, jyiy, j, xi, x, z)
    assert isinstance(result, dict)
