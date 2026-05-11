"""Tests for msm231.mvsml_ridge_lasso_elastic_eq_9_44."""
import numpy as np
import pytest
from morie.fn.msm231 import mvsml_ridge_lasso_elastic_eq_9_44


def test_msm231_basic():
    """Test basic functionality."""
    Wolfe = np.random.default_rng(42).normal(0, 1, 100)
    dual = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    maximization = np.random.default_rng(42).normal(0, 1, 100)
    problem = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_44(Wolfe, dual, version, maximization, problem, of)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm231_edge():
    """Test edge cases."""
    Wolfe = np.random.default_rng(42).normal(0, 1, 100)
    dual = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    maximization = np.random.default_rng(42).normal(0, 1, 100)
    problem = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_44(Wolfe, dual, version, maximization, problem, of)
    assert isinstance(result, dict)
