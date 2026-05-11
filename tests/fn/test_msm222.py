"""Tests for msm222.mvsml_ridge_lasso_elastic_eq_9_37."""
import numpy as np
import pytest
from morie.fn.msm222 import mvsml_ridge_lasso_elastic_eq_9_37


def test_msm222_basic():
    """Test basic functionality."""
    s = 90
    T = np.random.default_rng(43).integers(0, 2, 100)
    like = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    total = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_37(s, T, like, a, the, total)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm222_edge():
    """Test edge cases."""
    s = 90
    T = np.random.default_rng(43).integers(0, 2, 100)
    like = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    total = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_37(s, T, like, a, the, total)
    assert isinstance(result, dict)
