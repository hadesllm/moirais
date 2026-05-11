"""Tests for msm068.mvsml_bayesian_regression_eq_6_9."""
import numpy as np
import pytest
from morie.fn.msm068 import mvsml_bayesian_regression_eq_6_9


def test_msm068_basic():
    """Test basic functionality."""
    means = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = mvsml_bayesian_regression_eq_6_9(means, that, the, random, matrix, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm068_edge():
    """Test edge cases."""
    means = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    matrix = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = mvsml_bayesian_regression_eq_6_9(means, that, the, random, matrix, Z)
    assert isinstance(result, dict)
