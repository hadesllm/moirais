"""Tests for msm162.mvsml_ridge_lasso_elastic_eq_9_1."""
import numpy as np
import pytest
from morie.fn.msm162 import mvsml_ridge_lasso_elastic_eq_9_1


def test_msm162_basic():
    """Test basic functionality."""
    original = np.random.default_rng(42).normal(0, 1, 100)
    space = np.random.default_rng(42).normal(0, 1, 100)
    has = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    dimension = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_1(original, space, has, a, dimension, of)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm162_edge():
    """Test edge cases."""
    original = np.random.default_rng(42).normal(0, 1, 100)
    space = np.random.default_rng(42).normal(0, 1, 100)
    has = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    dimension = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_1(original, space, has, a, dimension, of)
    assert isinstance(result, dict)
