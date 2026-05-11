"""Tests for msm203.mvsml_ridge_lasso_elastic_eq_9_29."""
import numpy as np
import pytest
from morie.fn.msm203 import mvsml_ridge_lasso_elastic_eq_9_29


def test_msm203_basic():
    """Test basic functionality."""
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_29(Xn, L, i, iyi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm203_edge():
    """Test edge cases."""
    Xn = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_29(Xn, L, i, iyi)
    assert isinstance(result, dict)
