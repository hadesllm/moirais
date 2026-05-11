"""Tests for msm234.mvsml_ridge_lasso_elastic_eq_9_46."""
import numpy as np
import pytest
from morie.fn.msm234 import mvsml_ridge_lasso_elastic_eq_9_46


def test_msm234_basic():
    """Test basic functionality."""
    to = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    following = np.random.default_rng(42).normal(0, 1, 100)
    optimization = np.random.default_rng(42).normal(0, 1, 100)
    problem = np.random.default_rng(42).normal(0, 1, 100)
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_46(to, the, following, optimization, problem, Xn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm234_edge():
    """Test edge cases."""
    to = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    following = np.random.default_rng(42).normal(0, 1, 100)
    optimization = np.random.default_rng(42).normal(0, 1, 100)
    problem = np.random.default_rng(42).normal(0, 1, 100)
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_46(to, the, following, optimization, problem, Xn)
    assert isinstance(result, dict)
