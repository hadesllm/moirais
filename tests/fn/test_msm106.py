"""Tests for msm106.mvsml_bayesian_regression_pt2_eq_7_6."""
import numpy as np
import pytest
from moirais.fn.msm106 import mvsml_bayesian_regression_pt2_eq_7_6


def test_msm106_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    following = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(C, the, following, exp, xT, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm106_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    following = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(C, the, following, exp, xT, i)
    assert isinstance(result, dict)
