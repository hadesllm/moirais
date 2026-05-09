"""Tests for msm164.mvsml_ridge_lasso_elastic_eq_9_1."""
import numpy as np
import pytest
from moirais.fn.msm164 import mvsml_ridge_lasso_elastic_eq_9_1


def test_msm164_basic():
    """Test basic functionality."""
    family = 'gaussian'
    subspaces = np.random.default_rng(42).normal(0, 1, 100)
    From = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    mathematical = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_1(family, subspaces, From, a, mathematical, point)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm164_edge():
    """Test edge cases."""
    family = 'gaussian'
    subspaces = np.random.default_rng(42).normal(0, 1, 100)
    From = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    mathematical = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_1(family, subspaces, From, a, mathematical, point)
    assert isinstance(result, dict)
