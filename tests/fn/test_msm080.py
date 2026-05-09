"""Tests for msm080.mvsml_bayesian_regression_eq_6_11."""
import numpy as np
import pytest
from moirais.fn.msm080 import mvsml_bayesian_regression_eq_6_11


def test_msm080_basic():
    """Test basic functionality."""
    Return = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    step = np.random.default_rng(42).normal(0, 1, 100)
    terminate = np.random.default_rng(42).normal(0, 1, 100)
    when = np.random.default_rng(42).normal(0, 1, 100)
    chain = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_11(Return, to, step, terminate, when, chain)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm080_edge():
    """Test edge cases."""
    Return = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    step = np.random.default_rng(42).normal(0, 1, 100)
    terminate = np.random.default_rng(42).normal(0, 1, 100)
    when = np.random.default_rng(42).normal(0, 1, 100)
    chain = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_11(Return, to, step, terminate, when, chain)
    assert isinstance(result, dict)
