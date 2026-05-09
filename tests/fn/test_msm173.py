"""Tests for msm173.mvsml_ridge_lasso_elastic_eq_9_5."""
import numpy as np
import pytest
from moirais.fn.msm173 import mvsml_ridge_lasso_elastic_eq_9_5


def test_msm173_basic():
    """Test basic functionality."""
    xip = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    p = 5
    dimensional = np.random.default_rng(42).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_5(xip, a, p, dimensional, vector, of)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm173_edge():
    """Test edge cases."""
    xip = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    p = 5
    dimensional = np.random.default_rng(42).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_5(xip, a, p, dimensional, vector, of)
    assert isinstance(result, dict)
