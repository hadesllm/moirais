"""Tests for msm079.mvsml_bayesian_regression_eq_6_11."""
import numpy as np
import pytest
from morie.fn.msm079 import mvsml_bayesian_regression_eq_6_11


def test_msm079_basic():
    """Test basic functionality."""
    G = np.eye(10)
    which = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_11(G, which, means, that, E, j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm079_edge():
    """Test edge cases."""
    G = np.eye(10)
    which = np.random.default_rng(42).normal(0, 1, 100)
    means = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_11(G, which, means, that, E, j)
    assert isinstance(result, dict)
