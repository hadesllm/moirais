"""Tests for msm197.mvsml_ridge_lasso_elastic_eq_9_25."""
import numpy as np
import pytest
from morie.fn.msm197 import mvsml_ridge_lasso_elastic_eq_9_25


def test_msm197_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_25(y, The, last, version, of, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm197_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_25(y, The, last, version, of, the)
    assert isinstance(result, dict)
