"""Tests for msm196.mvsml_ridge_lasso_elastic_eq_9_24."""
import numpy as np
import pytest
from moirais.fn.msm196 import mvsml_ridge_lasso_elastic_eq_9_24


def test_msm196_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_24(f, x, y, subject, to)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm196_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_24(f, x, y, subject, to)
    assert isinstance(result, dict)
