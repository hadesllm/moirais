"""Tests for msm186.mvsml_ridge_lasso_elastic_eq_9_13."""
import numpy as np
import pytest
from morie.fn.msm186 import mvsml_ridge_lasso_elastic_eq_9_13


def test_msm186_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    Xm = np.random.default_rng(42).normal(0, 1, 100)
    Xp = np.random.default_rng(42).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_13(z, x, Xm, Xp, subject, to)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm186_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    Xm = np.random.default_rng(42).normal(0, 1, 100)
    Xp = np.random.default_rng(42).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_13(z, x, Xm, Xp, subject, to)
    assert isinstance(result, dict)
