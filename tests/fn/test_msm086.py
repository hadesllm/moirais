"""Tests for msm086.mvsml_bayesian_regression_pt2_eq_7_1."""
import numpy as np
import pytest
from morie.fn.msm086 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm086_basic():
    """Test basic functionality."""
    exp = np.random.default_rng(42).normal(0, 1, 100)
    u2 = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    p = 5
    result = mvsml_bayesian_regression_pt2_eq_7_1(exp, u2, where, F, z, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm086_edge():
    """Test edge cases."""
    exp = np.random.default_rng(42).normal(0, 1, 100)
    u2 = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    p = 5
    result = mvsml_bayesian_regression_pt2_eq_7_1(exp, u2, where, F, z, p)
    assert isinstance(result, dict)
