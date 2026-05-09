"""Tests for msm192.mvsml_ridge_lasso_elastic_eq_9_20."""
import numpy as np
import pytest
from moirais.fn.msm192 import mvsml_ridge_lasso_elastic_eq_9_20


def test_msm192_basic():
    """Test basic functionality."""
    Then = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    Wolfe = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_20(Then, the, last, version, of, Wolfe)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm192_edge():
    """Test edge cases."""
    Then = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    last = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    Wolfe = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_20(Then, the, last, version, of, Wolfe)
    assert isinstance(result, dict)
