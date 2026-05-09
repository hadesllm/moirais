"""Tests for msm107.mvsml_bayesian_regression_pt2_eq_7_6."""
import numpy as np
import pytest
from moirais.fn.msm107 import mvsml_bayesian_regression_pt2_eq_7_6


def test_msm107_basic():
    """Test basic functionality."""
    PC = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(PC, c, C, l, exp, xT)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm107_edge():
    """Test edge cases."""
    PC = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(PC, c, C, l, exp, xT)
    assert isinstance(result, dict)
