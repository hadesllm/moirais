"""Tests for msm048.mvsml_bayesian_regression_eq_6_3."""
import numpy as np
import pytest
from moirais.fn.msm048 import mvsml_bayesian_regression_eq_6_3


def test_msm048_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    X1XT = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    which = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_3(where, g, p, X1XT, G, which)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm048_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    X1XT = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    which = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_3(where, g, p, X1XT, G, which)
    assert isinstance(result, dict)
