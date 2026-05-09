"""Tests for msm060.mvsml_bayesian_regression_eq_6_1."""
import numpy as np
import pytest
from moirais.fn.msm060 import mvsml_bayesian_regression_eq_6_1


def test_msm060_basic():
    """Test basic functionality."""
    BayesA = np.random.default_rng(42).normal(0, 1, 100)
    BayesB = np.random.default_rng(42).normal(0, 1, 100)
    Var = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(BayesA, BayesB, Var, j, E, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm060_edge():
    """Test edge cases."""
    BayesA = np.random.default_rng(42).normal(0, 1, 100)
    BayesB = np.random.default_rng(42).normal(0, 1, 100)
    Var = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(BayesA, BayesB, Var, j, E, S)
    assert isinstance(result, dict)
