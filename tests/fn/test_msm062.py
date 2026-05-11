"""Tests for msm062.mvsml_bayesian_regression_eq_6_6."""
import numpy as np
import pytest
from morie.fn.msm062 import mvsml_bayesian_regression_eq_6_6


def test_msm062_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    XEM = np.random.default_rng(42).normal(0, 1, 100)
    EM = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_6(y, XE, E, X, XEM, EM)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm062_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    XE = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    XEM = np.random.default_rng(42).normal(0, 1, 100)
    EM = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_6(y, XE, E, X, XEM, EM)
    assert isinstance(result, dict)
