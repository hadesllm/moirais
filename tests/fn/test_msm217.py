"""Tests for msm217.mvsml_ridge_lasso_elastic_eq_9_33."""
import numpy as np
import pytest
from morie.fn.msm217 import mvsml_ridge_lasso_elastic_eq_9_33


def test_msm217_basic():
    """Test basic functionality."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    dot = np.random.default_rng(42).normal(0, 1, 100)
    product = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    vectors = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_33(e, dot, product, of, vectors, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm217_edge():
    """Test edge cases."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    dot = np.random.default_rng(42).normal(0, 1, 100)
    product = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    vectors = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_33(e, dot, product, of, vectors, xi)
    assert isinstance(result, dict)
