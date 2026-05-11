"""Tests for msm045.mvsml_bayesian_regression_eq_6_1."""
import numpy as np
import pytest
from morie.fn.msm045 import mvsml_bayesian_regression_eq_6_1


def test_msm045_basic():
    """Test basic functionality."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Genome = np.random.default_rng(42).normal(0, 1, 100)
    Based = np.random.default_rng(42).normal(0, 1, 100)
    Ridge = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    When = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(Bayesian, Genome, Based, Ridge, Regression, When)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm045_edge():
    """Test edge cases."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Genome = np.random.default_rng(42).normal(0, 1, 100)
    Based = np.random.default_rng(42).normal(0, 1, 100)
    Ridge = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    When = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(Bayesian, Genome, Based, Ridge, Regression, When)
    assert isinstance(result, dict)
