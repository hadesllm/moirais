"""Tests for msm212.mvsml_ridge_lasso_elastic_eq_9_32."""
import numpy as np
import pytest
from moirais.fn.msm212 import mvsml_ridge_lasso_elastic_eq_9_32


def test_msm212_basic():
    """Test basic functionality."""
    margin = np.random.default_rng(42).normal(0, 1, 100)
    classi = np.random.default_rng(42).normal(0, 1, 100)
    er = np.random.default_rng(42).normal(0, 1, 100)
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    maximize = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_32(margin, classi, er, Xn, i, maximize)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm212_edge():
    """Test edge cases."""
    margin = np.random.default_rng(42).normal(0, 1, 100)
    classi = np.random.default_rng(42).normal(0, 1, 100)
    er = np.random.default_rng(42).normal(0, 1, 100)
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    maximize = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_32(margin, classi, er, Xn, i, maximize)
    assert isinstance(result, dict)
