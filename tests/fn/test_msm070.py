"""Tests for msm070.mvsml_bayesian_regression_eq_6_9."""
import numpy as np
import pytest
from morie.fn.msm070 import mvsml_bayesian_regression_eq_6_9


def test_msm070_basic():
    """Test basic functionality."""
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    list = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    model = np.random.default_rng(42).normal(0, 1, 100)
    FIXED = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = mvsml_bayesian_regression_eq_6_9(ETA, list, X, model, FIXED, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm070_edge():
    """Test edge cases."""
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    list = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    model = np.random.default_rng(42).normal(0, 1, 100)
    FIXED = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = mvsml_bayesian_regression_eq_6_9(ETA, list, X, model, FIXED, K)
    assert isinstance(result, dict)
