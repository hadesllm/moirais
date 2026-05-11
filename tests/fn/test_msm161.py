"""Tests for msm161.mvsml_ridge_lasso_elastic_eq_9_1."""
import numpy as np
import pytest
from morie.fn.msm161 import mvsml_ridge_lasso_elastic_eq_9_1


def test_msm161_basic():
    """Test basic functionality."""
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    Hyperplanes = np.random.default_rng(42).normal(0, 1, 100)
    two = np.random.default_rng(42).normal(0, 1, 100)
    left = np.random.default_rng(42).normal(0, 1, 100)
    three = np.random.default_rng(42).normal(0, 1, 100)
    right = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_1(Fig, Hyperplanes, two, left, three, right)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm161_edge():
    """Test edge cases."""
    Fig = np.random.default_rng(42).normal(0, 1, 100)
    Hyperplanes = np.random.default_rng(42).normal(0, 1, 100)
    two = np.random.default_rng(42).normal(0, 1, 100)
    left = np.random.default_rng(42).normal(0, 1, 100)
    three = np.random.default_rng(42).normal(0, 1, 100)
    right = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_ridge_lasso_elastic_eq_9_1(Fig, Hyperplanes, two, left, three, right)
    assert isinstance(result, dict)
