"""Tests for msm046.mvsml_bayesian_regression_eq_6_3."""
import numpy as np
import pytest
from morie.fn.msm046 import mvsml_bayesian_regression_eq_6_3


def test_msm046_basic():
    """Test basic functionality."""
    non = np.random.default_rng(42).normal(0, 1, 100)
    informative = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    given = np.random.default_rng(42).normal(0, 1, 100)
    Christensen = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_3(non, informative, prior, given, Christensen, et)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm046_edge():
    """Test edge cases."""
    non = np.random.default_rng(42).normal(0, 1, 100)
    informative = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    given = np.random.default_rng(42).normal(0, 1, 100)
    Christensen = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_3(non, informative, prior, given, Christensen, et)
    assert isinstance(result, dict)
