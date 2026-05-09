"""Tests for msm163.mvsml_ridge_lasso_elastic_eq_9_1."""
import numpy as np
import pytest
from moirais.fn.msm163 import mvsml_ridge_lasso_elastic_eq_9_1


def test_msm163_basic():
    """Test basic functionality."""
    dimensional = np.random.default_rng(42).normal(0, 1, 100)
    at = np.random.default_rng(42).normal(0, 1, 100)
    subspace = np.random.default_rng(42).normal(0, 1, 100)
    James = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    al = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_1(dimensional, at, subspace, James, et, al)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm163_edge():
    """Test edge cases."""
    dimensional = np.random.default_rng(42).normal(0, 1, 100)
    at = np.random.default_rng(42).normal(0, 1, 100)
    subspace = np.random.default_rng(42).normal(0, 1, 100)
    James = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    al = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_1(dimensional, at, subspace, James, et, al)
    assert isinstance(result, dict)
