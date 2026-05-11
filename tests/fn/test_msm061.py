"""Tests for msm061.mvsml_bayesian_regression_eq_6_6."""
import numpy as np
import pytest
from morie.fn.msm061 import mvsml_bayesian_regression_eq_6_6


def test_msm061_basic():
    """Test basic functionality."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    inverse = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    parameter = np.random.default_rng(42).normal(0, 1, 100)
    any = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_6(j, inverse, of, the, parameter, any)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm061_edge():
    """Test edge cases."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    inverse = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    parameter = np.random.default_rng(42).normal(0, 1, 100)
    any = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_6(j, inverse, of, the, parameter, any)
    assert isinstance(result, dict)
