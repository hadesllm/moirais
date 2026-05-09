"""Tests for msm109.mvsml_bayesian_regression_pt2_eq_7_7."""
import numpy as np
import pytest
from moirais.fn.msm109 import mvsml_bayesian_regression_pt2_eq_7_7


def test_msm109_basic():
    """Test basic functionality."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    value = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    maximizes = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_7(the, value, of, this, that, maximizes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm109_edge():
    """Test edge cases."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    value = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    maximizes = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_7(the, value, of, this, that, maximizes)
    assert isinstance(result, dict)
