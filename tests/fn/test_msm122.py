"""Tests for msm122.mvsml_bayesian_regression_pt2_eq_7_11."""
import numpy as np
import pytest
from morie.fn.msm122 import mvsml_bayesian_regression_pt2_eq_7_11


def test_msm122_basic():
    """Test basic functionality."""
    Poisson = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    Given = np.random.default_rng(42).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_11(Poisson, regression, Given, vector, covariates, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm122_edge():
    """Test edge cases."""
    Poisson = np.random.default_rng(42).normal(0, 1, 100)
    regression = np.random.default_rng(42).normal(0, 1, 100)
    Given = np.random.default_rng(42).normal(0, 1, 100)
    vector = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_11(Poisson, regression, Given, vector, covariates, xi)
    assert isinstance(result, dict)
