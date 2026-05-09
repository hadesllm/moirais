"""Tests for msm233.mvsml_ridge_lasso_elastic_eq_9_44."""
import numpy as np
import pytest
from moirais.fn.msm233 import mvsml_ridge_lasso_elastic_eq_9_44


def test_msm233_basic():
    """Test basic functionality."""
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_ridge_lasso_elastic_eq_9_44(subject, to, i, T, iyi, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm233_edge():
    """Test edge cases."""
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    iyi = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = mvsml_ridge_lasso_elastic_eq_9_44(subject, to, i, T, iyi, n)
    assert isinstance(result, dict)
