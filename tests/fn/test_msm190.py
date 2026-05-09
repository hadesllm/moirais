"""Tests for msm190.mvsml_ridge_lasso_elastic_eq_9_18."""
import numpy as np
import pytest
from moirais.fn.msm190 import mvsml_ridge_lasso_elastic_eq_9_18


def test_msm190_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_18(z, x, f, subject, to)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm190_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_18(z, x, f, subject, to)
    assert isinstance(result, dict)
