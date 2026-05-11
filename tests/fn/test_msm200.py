"""Tests for msm200.mvsml_ridge_lasso_elastic_eq_9_8."""
import numpy as np
import pytest
from morie.fn.msm200 import mvsml_ridge_lasso_elastic_eq_9_8


def test_msm200_basic():
    """Test basic functionality."""
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    With = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_8(subject, to, With, this, last, version)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm200_edge():
    """Test edge cases."""
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    With = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_8(subject, to, With, this, last, version)
    assert isinstance(result, dict)
