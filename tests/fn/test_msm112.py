"""Tests for msm112.mvsml_bayesian_regression_pt2_eq_7_9."""
import numpy as np
import pytest
from moirais.fn.msm112 import mvsml_bayesian_regression_pt2_eq_7_9


def test_msm112_basic():
    """Test basic functionality."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    That = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    update = np.random.default_rng(42).normal(0, 1, 100)
    block = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_9(of, e, That, the, update, block)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm112_edge():
    """Test edge cases."""
    of = np.random.default_rng(42).normal(0, 1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    That = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    update = np.random.default_rng(42).normal(0, 1, 100)
    block = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_9(of, e, That, the, update, block)
    assert isinstance(result, dict)
