"""Tests for msm202.mvsml_ridge_lasso_elastic_eq_9_28."""
import numpy as np
import pytest
from morie.fn.msm202 import mvsml_ridge_lasso_elastic_eq_9_28


def test_msm202_basic():
    """Test basic functionality."""
    n = 100
    are = np.random.default_rng(42).normal(0, 1, 100)
    called = np.random.default_rng(42).normal(0, 1, 100)
    Lagrange = np.random.default_rng(42).normal(0, 1, 100)
    multipliers = np.random.default_rng(42).normal(0, 1, 100)
    Setting = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_28(n, are, called, Lagrange, multipliers, Setting)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm202_edge():
    """Test edge cases."""
    n = 100
    are = np.random.default_rng(42).normal(0, 1, 100)
    called = np.random.default_rng(42).normal(0, 1, 100)
    Lagrange = np.random.default_rng(42).normal(0, 1, 100)
    multipliers = np.random.default_rng(42).normal(0, 1, 100)
    Setting = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_28(n, are, called, Lagrange, multipliers, Setting)
    assert isinstance(result, dict)
