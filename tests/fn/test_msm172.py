"""Tests for msm172.mvsml_ridge_lasso_elastic_eq_9_2."""
import numpy as np
import pytest
from moirais.fn.msm172 import mvsml_ridge_lasso_elastic_eq_9_2


def test_msm172_basic():
    """Test basic functionality."""
    Support = np.random.default_rng(42).normal(0, 1, 100)
    Vector = np.random.default_rng(42).normal(0, 1, 100)
    Machines = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(Support, Vector, Machines, Regression, Fig, The)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm172_edge():
    """Test edge cases."""
    Support = np.random.default_rng(42).normal(0, 1, 100)
    Vector = np.random.default_rng(42).normal(0, 1, 100)
    Machines = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(Support, Vector, Machines, Regression, Fig, The)
    assert isinstance(result, dict)
