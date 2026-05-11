"""Tests for msm104.mvsml_bayesian_regression_pt2_eq_7_1."""
import numpy as np
import pytest
from morie.fn.msm104 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm104_basic():
    """Test basic functionality."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    PCCC = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    second = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    was = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(the, PCCC, of, second, model, was)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm104_edge():
    """Test edge cases."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    PCCC = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    second = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    was = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(the, PCCC, of, second, model, was)
    assert isinstance(result, dict)
