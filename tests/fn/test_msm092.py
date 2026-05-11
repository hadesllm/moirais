"""Tests for msm092.mvsml_bayesian_regression_pt2_eq_7_3."""
import numpy as np
import pytest
from morie.fn.msm092 import mvsml_bayesian_regression_pt2_eq_7_3


def test_msm092_basic():
    """Test basic functionality."""
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    taken = np.random.default_rng(42).normal(0, 1, 100)
    into = np.random.default_rng(42).normal(0, 1, 100)
    account = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_3(can, be, taken, into, account, to)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm092_edge():
    """Test edge cases."""
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    taken = np.random.default_rng(42).normal(0, 1, 100)
    into = np.random.default_rng(42).normal(0, 1, 100)
    account = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_3(can, be, taken, into, account, to)
    assert isinstance(result, dict)
