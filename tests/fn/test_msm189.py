"""Tests for msm189.mvsml_ridge_lasso_elastic_eq_9_17."""
import numpy as np
import pytest
from morie.fn.msm189 import mvsml_ridge_lasso_elastic_eq_9_17


def test_msm189_basic():
    """Test basic functionality."""
    Its = np.random.default_rng(42).normal(0, 1, 100)
    dual = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    according = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    Wolfe = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_17(Its, dual, version, according, to, Wolfe)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm189_edge():
    """Test edge cases."""
    Its = np.random.default_rng(42).normal(0, 1, 100)
    dual = np.random.default_rng(42).normal(0, 1, 100)
    version = np.random.default_rng(42).normal(0, 1, 100)
    according = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    Wolfe = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_17(Its, dual, version, according, to, Wolfe)
    assert isinstance(result, dict)
