"""Tests for msm177.mvsml_ridge_lasso_elastic_eq_9_5."""
import numpy as np
import pytest
from morie.fn.msm177 import mvsml_ridge_lasso_elastic_eq_9_5


def test_msm177_basic():
    """Test basic functionality."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    term = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_5(The, term, yi, xT, i, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm177_edge():
    """Test edge cases."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    term = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_5(The, term, yi, xT, i, the)
    assert isinstance(result, dict)
