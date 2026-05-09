"""Tests for msm072.mvsml_bayesian_regression_eq_6_10."""
import numpy as np
import pytest
from moirais.fn.msm072 import mvsml_bayesian_regression_eq_6_10


def test_msm072_basic():
    """Test basic functionality."""
    respectively = np.random.default_rng(42).normal(0, 1, 100)
    In = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    third = np.random.default_rng(42).normal(0, 1, 100)
    argument = np.random.default_rng(42).normal(0, 1, 100)
    resCOV = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_10(respectively, In, the, third, argument, resCOV)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm072_edge():
    """Test edge cases."""
    respectively = np.random.default_rng(42).normal(0, 1, 100)
    In = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    third = np.random.default_rng(42).normal(0, 1, 100)
    argument = np.random.default_rng(42).normal(0, 1, 100)
    resCOV = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_10(respectively, In, the, third, argument, resCOV)
    assert isinstance(result, dict)
