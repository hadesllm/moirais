"""Tests for msm043.mvsml_bayesian_regression_eq_6_2."""
import numpy as np
import pytest
from morie.fn.msm043 import mvsml_bayesian_regression_eq_6_2


def test_msm043_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    j = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    error = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_2(X, j, E, a, random, error)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm043_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    j = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    error = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_2(X, j, E, a, random, error)
    assert isinstance(result, dict)
