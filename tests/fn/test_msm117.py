"""Tests for msm117.mvsml_bayesian_regression_pt2_eq_7_7."""
import numpy as np
import pytest
from moirais.fn.msm117 import mvsml_bayesian_regression_pt2_eq_7_7


def test_msm117_basic():
    """Test basic functionality."""
    Like = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    penalized = np.random.default_rng(42).normal(0, 1, 100)
    logistic = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    studied = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_7(Like, the, penalized, logistic, regression, studied)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm117_edge():
    """Test edge cases."""
    Like = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    penalized = np.random.default_rng(42).normal(0, 1, 100)
    logistic = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    studied = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_7(Like, the, penalized, logistic, regression, studied)
    assert isinstance(result, dict)
