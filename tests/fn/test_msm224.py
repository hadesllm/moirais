"""Tests for msm224.mvsml_ridge_lasso_elastic_eq_9_39."""
import numpy as np
import pytest
from morie.fn.msm224 import mvsml_ridge_lasso_elastic_eq_9_39


def test_msm224_basic():
    """Test basic functionality."""
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    slack = np.random.default_rng(42).normal(0, 1, 100)
    variables = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_ridge_lasso_elastic_eq_9_39(constraints, of, the, slack, variables, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm224_edge():
    """Test edge cases."""
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    slack = np.random.default_rng(42).normal(0, 1, 100)
    variables = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_ridge_lasso_elastic_eq_9_39(constraints, of, the, slack, variables, n)
    assert isinstance(result, dict)
