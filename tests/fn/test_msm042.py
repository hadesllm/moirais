"""Tests for msm042.mvsml_bayesian_regression_eq_6_1."""
import numpy as np
import pytest
from morie.fn.msm042 import mvsml_bayesian_regression_eq_6_1


def test_msm042_basic():
    """Test basic functionality."""
    interest = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    non = np.random.default_rng(42).normal(0, 1, 100)
    phenotyped = np.random.default_rng(42).normal(0, 1, 100)
    individuals = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(interest, of, the, non, phenotyped, individuals)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm042_edge():
    """Test edge cases."""
    interest = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    non = np.random.default_rng(42).normal(0, 1, 100)
    phenotyped = np.random.default_rng(42).normal(0, 1, 100)
    individuals = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(interest, of, the, non, phenotyped, individuals)
    assert isinstance(result, dict)
