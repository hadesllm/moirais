"""Tests for msm174.mvsml_ridge_lasso_elastic_eq_9_5."""
import numpy as np
import pytest
from morie.fn.msm174 import mvsml_ridge_lasso_elastic_eq_9_5


def test_msm174_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    positive = np.random.default_rng(42).normal(0, 1, 100)
    labeled = np.random.default_rng(42).normal(0, 1, 100)
    bf = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_5(i, b, positive, labeled, bf, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm174_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    positive = np.random.default_rng(42).normal(0, 1, 100)
    labeled = np.random.default_rng(42).normal(0, 1, 100)
    bf = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_5(i, b, positive, labeled, bf, xi)
    assert isinstance(result, dict)
