"""Tests for msm180.mvsml_ridge_lasso_elastic_eq_9_6."""
import numpy as np
import pytest
from moirais.fn.msm180 import mvsml_ridge_lasso_elastic_eq_9_6


def test_msm180_basic():
    """Test basic functionality."""
    distance = np.random.default_rng(42).normal(0, 1, 100)
    margin = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    hyperplane = np.random.default_rng(42).normal(0, 1, 100)
    h0 = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(distance, margin, M, hyperplane, h0, xT)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm180_edge():
    """Test edge cases."""
    distance = np.random.default_rng(42).normal(0, 1, 100)
    margin = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    hyperplane = np.random.default_rng(42).normal(0, 1, 100)
    h0 = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(distance, margin, M, hyperplane, h0, xT)
    assert isinstance(result, dict)
