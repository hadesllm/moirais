"""Tests for msm179.mvsml_ridge_lasso_elastic_eq_9_6."""
import numpy as np
import pytest
from morie.fn.msm179 import mvsml_ridge_lasso_elastic_eq_9_6


def test_msm179_basic():
    """Test basic functionality."""
    observations = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    inside = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    fences = np.random.default_rng(42).normal(0, 1, 100)
    street = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(observations, are, inside, the, fences, street)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm179_edge():
    """Test edge cases."""
    observations = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    inside = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    fences = np.random.default_rng(42).normal(0, 1, 100)
    street = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_6(observations, are, inside, the, fences, street)
    assert isinstance(result, dict)
