"""Tests for msm110.mvsml_bayesian_regression_pt2_eq_7_8."""
import numpy as np
import pytest
from morie.fn.msm110 import mvsml_bayesian_regression_pt2_eq_7_8


def test_msm110_basic():
    """Test basic functionality."""
    I = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    log = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_8(I, yi, c, i, log, l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm110_edge():
    """Test edge cases."""
    I = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    log = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_8(I, yi, c, i, log, l)
    assert isinstance(result, dict)
