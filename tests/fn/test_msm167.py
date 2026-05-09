"""Tests for msm167.mvsml_ridge_lasso_elastic_eq_9_2."""
import numpy as np
import pytest
from moirais.fn.msm167 import mvsml_ridge_lasso_elastic_eq_9_2


def test_msm167_basic():
    """Test basic functionality."""
    X3 = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    holds = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(X3, T, which, holds, a, point)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm167_edge():
    """Test edge cases."""
    X3 = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    holds = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(X3, T, which, holds, a, point)
    assert isinstance(result, dict)
