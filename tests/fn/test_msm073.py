"""Tests for msm073.mvsml_bayesian_regression_eq_6_8."""
import numpy as np
import pytest
from morie.fn.msm073 import mvsml_bayesian_regression_eq_6_8


def test_msm073_basic():
    """Test basic functionality."""
    implemented = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    multivariate = np.random.default_rng(42).normal(0, 1, 100)
    Ridge = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_8(implemented, a, multivariate, Ridge, regression, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm073_edge():
    """Test edge cases."""
    implemented = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    multivariate = np.random.default_rng(42).normal(0, 1, 100)
    Ridge = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_8(implemented, a, multivariate, Ridge, regression, model)
    assert isinstance(result, dict)
