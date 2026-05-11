"""Tests for msm182.mvsml_ridge_lasso_elastic_eq_9_7."""
import numpy as np
import pytest
from morie.fn.msm182 import mvsml_ridge_lasso_elastic_eq_9_7


def test_msm182_basic():
    """Test basic functionality."""
    increasing = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    we = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    now = np.random.default_rng(42).normal(0, 1, 100)
    reformulate = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_7(increasing, k, we, can, now, reformulate)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm182_edge():
    """Test edge cases."""
    increasing = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    we = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    now = np.random.default_rng(42).normal(0, 1, 100)
    reformulate = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_7(increasing, k, we, can, now, reformulate)
    assert isinstance(result, dict)
