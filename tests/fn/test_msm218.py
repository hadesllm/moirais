"""Tests for msm218.mvsml_ridge_lasso_elastic_eq_9_34."""
import numpy as np
import pytest
from morie.fn.msm218 import mvsml_ridge_lasso_elastic_eq_9_34


def test_msm218_basic():
    """Test basic functionality."""
    choosing = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    right = np.random.default_rng(42).normal(0, 1, 100)
    hyperplane = np.random.default_rng(42).normal(0, 1, 100)
    we = np.random.default_rng(42).normal(0, 1, 100)
    need = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_34(choosing, the, right, hyperplane, we, need)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm218_edge():
    """Test edge cases."""
    choosing = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    right = np.random.default_rng(42).normal(0, 1, 100)
    hyperplane = np.random.default_rng(42).normal(0, 1, 100)
    we = np.random.default_rng(42).normal(0, 1, 100)
    need = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_34(choosing, the, right, hyperplane, we, need)
    assert isinstance(result, dict)
