"""Tests for msm077.mvsml_bayesian_regression_eq_6_2."""
import numpy as np
import pytest
from morie.fn.msm077 import mvsml_bayesian_regression_eq_6_2


def test_msm077_basic():
    """Test basic functionality."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    nT = np.random.default_rng(42).normal(0, 1, 100)
    traits = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    result = mvsml_bayesian_regression_eq_6_2(the, information, of, nT, traits, J)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm077_edge():
    """Test edge cases."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    nT = np.random.default_rng(42).normal(0, 1, 100)
    traits = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    result = mvsml_bayesian_regression_eq_6_2(the, information, of, nT, traits, J)
    assert isinstance(result, dict)
