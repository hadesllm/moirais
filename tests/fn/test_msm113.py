"""Tests for msm113.mvsml_bayesian_regression_pt2_eq_7_9."""
import numpy as np
import pytest
from morie.fn.msm113 import mvsml_bayesian_regression_pt2_eq_7_9


def test_msm113_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    TWcX = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_9(T, b, c, X, TWcX, D)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm113_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    TWcX = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_9(T, b, c, X, TWcX, D)
    assert isinstance(result, dict)
