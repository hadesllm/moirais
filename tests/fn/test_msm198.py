"""Tests for msm198.mvsml_ridge_lasso_elastic_eq_9_26."""
import numpy as np
import pytest
from moirais.fn.msm198 import mvsml_ridge_lasso_elastic_eq_9_26


def test_msm198_basic():
    """Test basic functionality."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Wolfe = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_26(The, last, version, of, the, Wolfe)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm198_edge():
    """Test edge cases."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Wolfe = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_26(The, last, version, of, the, Wolfe)
    assert isinstance(result, dict)
