"""Tests for msm210.mvsml_ridge_lasso_elastic_eq_9_31."""
import numpy as np
import pytest
from morie.fn.msm210 import mvsml_ridge_lasso_elastic_eq_9_31


def test_msm210_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_31(i, jyiy, j, xi, x, iyi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm210_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_31(i, jyiy, j, xi, x, iyi)
    assert isinstance(result, dict)
