"""Tests for msm049.mvsml_bayesian_regression_eq_6_4."""
import numpy as np
import pytest
from moirais.fn.msm049 import mvsml_bayesian_regression_eq_6_4


def test_msm049_basic():
    """Test basic functionality."""
    p = 5
    X1XT = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    which = np.random.default_rng(42).normal(0, 1, 100)
    known = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(p, X1XT, G, which, known, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm049_edge():
    """Test edge cases."""
    p = 5
    X1XT = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    which = np.random.default_rng(42).normal(0, 1, 100)
    known = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(p, X1XT, G, which, known, the)
    assert isinstance(result, dict)
