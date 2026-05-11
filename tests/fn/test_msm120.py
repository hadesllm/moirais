"""Tests for msm120.mvsml_bayesian_regression_pt2_eq_7_6."""
import numpy as np
import pytest
from morie.fn.msm120 import mvsml_bayesian_regression_pt2_eq_7_6


def test_msm120_basic():
    """Test basic functionality."""
    Lasso = np.random.default_rng(42).normal(0, 1, 100)
    penalization = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    same = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(Lasso, penalization, models, are, the, same)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm120_edge():
    """Test edge cases."""
    Lasso = np.random.default_rng(42).normal(0, 1, 100)
    penalization = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    same = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(Lasso, penalization, models, are, the, same)
    assert isinstance(result, dict)
