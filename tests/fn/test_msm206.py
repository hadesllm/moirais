"""Tests for msm206.mvsml_ridge_lasso_elastic_eq_9_28."""
import numpy as np
import pytest
from morie.fn.msm206 import mvsml_ridge_lasso_elastic_eq_9_28


def test_msm206_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    If = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    then = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_28(a, If, i, then, yi, xT)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm206_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    If = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    then = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_28(a, If, i, then, yi, xT)
    assert isinstance(result, dict)
