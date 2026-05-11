"""Tests for msm195.mvsml_ridge_lasso_elastic_eq_9_23."""
import numpy as np
import pytest
from morie.fn.msm195 import mvsml_ridge_lasso_elastic_eq_9_23


def test_msm195_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    Its = np.random.default_rng(42).normal(0, 1, 100)
    dual = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_23(x, y, subject, to, Its, dual)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm195_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    Its = np.random.default_rng(42).normal(0, 1, 100)
    dual = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_23(x, y, subject, to, Its, dual)
    assert isinstance(result, dict)
