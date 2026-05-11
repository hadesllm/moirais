"""Tests for msm168.mvsml_ridge_lasso_elastic_eq_9_2."""
import numpy as np
import pytest
from morie.fn.msm168 import mvsml_ridge_lasso_elastic_eq_9_2


def test_msm168_basic():
    """Test basic functionality."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    plane = np.random.default_rng(42).normal(0, 1, 100)
    since = np.random.default_rng(42).normal(0, 1, 100)
    three = np.random.default_rng(42).normal(0, 1, 100)
    dimensions = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(of, a, plane, since, three, dimensions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm168_edge():
    """Test edge cases."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    plane = np.random.default_rng(42).normal(0, 1, 100)
    since = np.random.default_rng(42).normal(0, 1, 100)
    three = np.random.default_rng(42).normal(0, 1, 100)
    dimensions = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_2(of, a, plane, since, three, dimensions)
    assert isinstance(result, dict)
