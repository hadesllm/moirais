"""Tests for msm094.mvsml_bayesian_regression_pt2_eq_7_1."""
import numpy as np
import pytest
from moirais.fn.msm094 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm094_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    BGLR = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ordinal = np.random.default_rng(42).normal(0, 1, 100)
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    nIter = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(A, BGLR, y, ordinal, ETA, nIter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm094_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    BGLR = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ordinal = np.random.default_rng(42).normal(0, 1, 100)
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    nIter = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(A, BGLR, y, ordinal, ETA, nIter)
    assert isinstance(result, dict)
