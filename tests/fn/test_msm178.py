"""Tests for msm178.mvsml_ridge_lasso_elastic_eq_9_6."""
import numpy as np
import pytest
from moirais.fn.msm178 import mvsml_ridge_lasso_elastic_eq_9_6


def test_msm178_basic():
    """Test basic functionality."""
    n = 100
    the = np.random.default_rng(42).normal(0, 1, 100)
    restrictions = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    optimization = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(n, the, restrictions, of, this, optimization)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm178_edge():
    """Test edge cases."""
    n = 100
    the = np.random.default_rng(42).normal(0, 1, 100)
    restrictions = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    optimization = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(n, the, restrictions, of, this, optimization)
    assert isinstance(result, dict)
