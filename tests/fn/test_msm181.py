"""Tests for msm181.mvsml_ridge_lasso_elastic_eq_9_6."""
import numpy as np
import pytest
from moirais.fn.msm181 import mvsml_ridge_lasso_elastic_eq_9_6


def test_msm181_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    total = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(i, This, means, that, the, total)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm181_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    This = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    total = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(i, This, means, that, the, total)
    assert isinstance(result, dict)
