"""Tests for msm191.mvsml_ridge_lasso_elastic_eq_9_19."""
import numpy as np
import pytest
from morie.fn.msm191 import mvsml_ridge_lasso_elastic_eq_9_19


def test_msm191_basic():
    """Test basic functionality."""
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    Then = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_19(subject, to, x, Then, the, last)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm191_edge():
    """Test edge cases."""
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    Then = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_19(subject, to, x, Then, the, last)
    assert isinstance(result, dict)
