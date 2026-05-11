"""Tests for msm100.mvsml_bayesian_regression_pt2_eq_7_4."""
import numpy as np
import pytest
from morie.fn.msm100 import mvsml_bayesian_regression_pt2_eq_7_4


def test_msm100_basic():
    """Test basic functionality."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    lines = np.random.default_rng(42).normal(0, 1, 100)
    So = np.random.default_rng(42).normal(0, 1, 100)
    only = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    M3 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_4(the, lines, So, only, models, M3)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm100_edge():
    """Test edge cases."""
    the = np.random.default_rng(42).normal(0, 1, 100)
    lines = np.random.default_rng(42).normal(0, 1, 100)
    So = np.random.default_rng(42).normal(0, 1, 100)
    only = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    M3 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_4(the, lines, So, only, models, M3)
    assert isinstance(result, dict)
