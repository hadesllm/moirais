"""Tests for msm170.mvsml_ridge_lasso_elastic_eq_9_3."""
import numpy as np
import pytest
from morie.fn.msm170 import mvsml_ridge_lasso_elastic_eq_9_3


def test_msm170_basic():
    """Test basic functionality."""
    it = np.random.default_rng(42).normal(0, 1, 100)
    simply = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    extension = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    pXp = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_3(it, simply, an, extension, of, pXp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm170_edge():
    """Test edge cases."""
    it = np.random.default_rng(42).normal(0, 1, 100)
    simply = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    extension = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    pXp = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_3(it, simply, an, extension, of, pXp)
    assert isinstance(result, dict)
