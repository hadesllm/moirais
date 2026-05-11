"""Tests for msm059.mvsml_bayesian_regression_eq_6_1."""
import numpy as np
import pytest
from morie.fn.msm059 import mvsml_bayesian_regression_eq_6_1


def test_msm059_basic():
    """Test basic functionality."""
    BGLR = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    nIter = np.random.default_rng(42).normal(0, 1, 100)
    burnIn = np.random.default_rng(42).normal(0, 1, 100)
    df0 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(BGLR, y, ETA, nIter, burnIn, df0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm059_edge():
    """Test edge cases."""
    BGLR = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    nIter = np.random.default_rng(42).normal(0, 1, 100)
    burnIn = np.random.default_rng(42).normal(0, 1, 100)
    df0 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(BGLR, y, ETA, nIter, burnIn, df0)
    assert isinstance(result, dict)
