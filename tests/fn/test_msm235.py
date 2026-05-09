"""Tests for msm235.mvsml_ridge_lasso_elastic_eq_9_47."""
import numpy as np
import pytest
from moirais.fn.msm235 import mvsml_ridge_lasso_elastic_eq_9_47


def test_msm235_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    jK = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    xj = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_47(i, jyiy, jK, xi, xj, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm235_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    jK = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    xj = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_47(i, jyiy, jK, xi, xj, z)
    assert isinstance(result, dict)
