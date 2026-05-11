"""Tests for msm121.mvsml_bayesian_regression_pt2_eq_7_6."""
import numpy as np
import pytest
from morie.fn.msm121 import mvsml_bayesian_regression_pt2_eq_7_6


def test_msm121_basic():
    """Test basic functionality."""
    tation = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    following = np.random.default_rng(42).normal(0, 1, 100)
    six = np.random.default_rng(42).normal(0, 1, 100)
    kinds = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(tation, of, the, following, six, kinds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm121_edge():
    """Test edge cases."""
    tation = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    following = np.random.default_rng(42).normal(0, 1, 100)
    six = np.random.default_rng(42).normal(0, 1, 100)
    kinds = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_6(tation, of, the, following, six, kinds)
    assert isinstance(result, dict)
