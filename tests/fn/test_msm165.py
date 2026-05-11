"""Tests for msm165.mvsml_ridge_lasso_elastic_eq_9_2."""
import numpy as np
import pytest
from morie.fn.msm165 import mvsml_ridge_lasso_elastic_eq_9_2


def test_msm165_basic():
    """Test basic functionality."""
    From = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    mathematical = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    view = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(From, a, mathematical, point, of, view)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm165_edge():
    """Test edge cases."""
    From = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    mathematical = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    view = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(From, a, mathematical, point, of, view)
    assert isinstance(result, dict)
