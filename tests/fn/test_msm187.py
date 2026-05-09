"""Tests for msm187.mvsml_ridge_lasso_elastic_eq_9_14."""
import numpy as np
import pytest
from moirais.fn.msm187 import mvsml_ridge_lasso_elastic_eq_9_14


def test_msm187_basic():
    """Test basic functionality."""
    Xm = np.random.default_rng(42).normal(0, 1, 100)
    Xp = np.random.default_rng(42).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_14(Xm, Xp, subject, to, f, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm187_edge():
    """Test edge cases."""
    Xm = np.random.default_rng(42).normal(0, 1, 100)
    Xp = np.random.default_rng(42).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_14(Xm, Xp, subject, to, f, x)
    assert isinstance(result, dict)
