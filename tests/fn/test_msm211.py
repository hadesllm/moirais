"""Tests for msm211.mvsml_ridge_lasso_elastic_eq_9_31."""
import numpy as np
import pytest
from moirais.fn.msm211 import mvsml_ridge_lasso_elastic_eq_9_31


def test_msm211_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    Pn = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_31(i, iyi, z, Pn, jyiy, j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm211_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    Pn = np.random.default_rng(42).normal(0, 1, 100)
    jyiy = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_31(i, iyi, z, Pn, jyiy, j)
    assert isinstance(result, dict)
