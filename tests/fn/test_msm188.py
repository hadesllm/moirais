"""Tests for msm188.mvsml_ridge_lasso_elastic_eq_9_15."""
import numpy as np
import pytest
from morie.fn.msm188 import mvsml_ridge_lasso_elastic_eq_9_15


def test_msm188_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    This = np.random.default_rng(42).normal(0, 1, 100)
    changes = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    searching = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_15(i, p, This, changes, the, searching)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm188_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    This = np.random.default_rng(42).normal(0, 1, 100)
    changes = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    searching = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_15(i, p, This, changes, the, searching)
    assert isinstance(result, dict)
