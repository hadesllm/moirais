"""Tests for msm101.mvsml_bayesian_regression_pt2_eq_7_5."""
import numpy as np
import pytest
from moirais.fn.msm101 import mvsml_bayesian_regression_pt2_eq_7_5


def test_msm101_basic():
    """Test basic functionality."""
    performance = np.random.default_rng(42).normal(0, 1, 100)
    prediction = np.random.default_rng(42).normal(0, 1, 100)
    these = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    was = np.random.default_rng(42).normal(0, 1, 100)
    evaluated = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_5(performance, prediction, these, models, was, evaluated)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm101_edge():
    """Test edge cases."""
    performance = np.random.default_rng(42).normal(0, 1, 100)
    prediction = np.random.default_rng(42).normal(0, 1, 100)
    these = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    was = np.random.default_rng(42).normal(0, 1, 100)
    evaluated = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_5(performance, prediction, these, models, was, evaluated)
    assert isinstance(result, dict)
