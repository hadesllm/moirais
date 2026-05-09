"""Tests for msm184.mvsml_ridge_lasso_elastic_eq_9_9."""
import numpy as np
import pytest
from moirais.fn.msm184 import mvsml_ridge_lasso_elastic_eq_9_9


def test_msm184_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    dual = np.random.default_rng(42).normal(0, 1, 100)
    result = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    explained = np.random.default_rng(42).normal(0, 1, 100)
    below = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_9(T, dual, result, which, explained, below)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm184_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    dual = np.random.default_rng(42).normal(0, 1, 100)
    result = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    explained = np.random.default_rng(42).normal(0, 1, 100)
    below = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_9(T, dual, result, which, explained, below)
    assert isinstance(result, dict)
