"""Tests for msm069.mvsml_bayesian_regression_eq_6_8."""
import numpy as np
import pytest
from morie.fn.msm069 import mvsml_bayesian_regression_eq_6_8


def test_msm069_basic():
    """Test basic functionality."""
    An = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_8(An, of, this, model, can, be)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm069_edge():
    """Test edge cases."""
    An = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_8(An, of, this, model, can, be)
    assert isinstance(result, dict)
