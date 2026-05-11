"""Tests for msm207.mvsml_ridge_lasso_elastic_eq_9_30."""
import numpy as np
import pytest
from morie.fn.msm207 import mvsml_ridge_lasso_elastic_eq_9_30


def test_msm207_basic():
    """Test basic functionality."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    slab = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    If = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_30(the, slab, b, If, yi, xT)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm207_edge():
    """Test edge cases."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    slab = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    If = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_30(the, slab, b, If, yi, xT)
    assert isinstance(result, dict)
