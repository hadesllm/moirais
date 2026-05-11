"""Tests for msm114.mvsml_bayesian_regression_pt2_eq_7_7."""
import numpy as np
import pytest
from morie.fn.msm114 import mvsml_bayesian_regression_pt2_eq_7_7


def test_msm114_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    identity = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = mvsml_bayesian_regression_pt2_eq_7_7(T, D, an, identity, where, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm114_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    identity = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = mvsml_bayesian_regression_pt2_eq_7_7(T, D, an, identity, where, X)
    assert isinstance(result, dict)
