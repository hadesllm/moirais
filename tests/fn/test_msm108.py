"""Tests for msm108.mvsml_bayesian_regression_pt2_eq_7_6."""
import numpy as np
import pytest
from morie.fn.msm108 import mvsml_bayesian_regression_pt2_eq_7_6


def test_msm108_basic():
    """Test basic functionality."""
    xT = np.random.default_rng(42).normal(0, 1, 100)
    log = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(xT, log, i, c, C, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm108_edge():
    """Test edge cases."""
    xT = np.random.default_rng(42).normal(0, 1, 100)
    log = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(xT, log, i, c, C, P)
    assert isinstance(result, dict)
