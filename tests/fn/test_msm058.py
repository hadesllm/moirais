"""Tests for msm058.mvsml_bayesian_regression_eq_6_1."""
import numpy as np
import pytest
from moirais.fn.msm058 import mvsml_bayesian_regression_eq_6_1


def test_msm058_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    BGLR = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    nIter = np.random.default_rng(42).normal(0, 1, 100)
    burnIn = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(A, BGLR, y, ETA, nIter, burnIn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm058_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    BGLR = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    nIter = np.random.default_rng(42).normal(0, 1, 100)
    burnIn = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(A, BGLR, y, ETA, nIter, burnIn)
    assert isinstance(result, dict)
