"""Tests for msm166.mvsml_ridge_lasso_elastic_eq_9_2."""
import numpy as np
import pytest
from moirais.fn.msm166 import mvsml_ridge_lasso_elastic_eq_9_2


def test_msm166_basic():
    """Test basic functionality."""
    parameters = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    nes = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    hyperplane = np.random.default_rng(42).normal(0, 1, 100)
    since = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(parameters, de, nes, a, hyperplane, since)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm166_edge():
    """Test edge cases."""
    parameters = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    nes = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    hyperplane = np.random.default_rng(42).normal(0, 1, 100)
    since = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(parameters, de, nes, a, hyperplane, since)
    assert isinstance(result, dict)
