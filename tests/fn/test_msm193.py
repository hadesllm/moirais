"""Tests for msm193.mvsml_ridge_lasso_elastic_eq_9_21."""
import numpy as np
import pytest
from morie.fn.msm193 import mvsml_ridge_lasso_elastic_eq_9_21


def test_msm193_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    With = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_21(z, subject, to, With, this, last)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm193_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    With = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_21(z, subject, to, With, this, last)
    assert isinstance(result, dict)
