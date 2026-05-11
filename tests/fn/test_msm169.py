"""Tests for msm169.mvsml_ridge_lasso_elastic_eq_9_3."""
import numpy as np
import pytest
from morie.fn.msm169 import mvsml_ridge_lasso_elastic_eq_9_3


def test_msm169_basic():
    """Test basic functionality."""
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    right = np.random.default_rng(42).normal(0, 1, 100)
    For = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_3(can, be, observed, Fig, right, For)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm169_edge():
    """Test edge cases."""
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    right = np.random.default_rng(42).normal(0, 1, 100)
    For = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_3(can, be, observed, Fig, right, For)
    assert isinstance(result, dict)
