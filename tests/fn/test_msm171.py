"""Tests for msm171.mvsml_ridge_lasso_elastic_eq_9_4."""
import numpy as np
import pytest
from morie.fn.msm171 import mvsml_ridge_lasso_elastic_eq_9_4


def test_msm171_basic():
    """Test basic functionality."""
    There = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    points = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    satisfying = np.random.default_rng(42).normal(0, 1, 100)
    lie = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_4(There, are, points, that, satisfying, lie)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm171_edge():
    """Test edge cases."""
    There = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    points = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    satisfying = np.random.default_rng(42).normal(0, 1, 100)
    lie = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_4(There, are, points, that, satisfying, lie)
    assert isinstance(result, dict)
