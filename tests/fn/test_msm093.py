"""Tests for msm093.mvsml_bayesian_regression_pt2_eq_7_3."""
import numpy as np
import pytest
from moirais.fn.msm093 import mvsml_bayesian_regression_pt2_eq_7_3


def test_msm093_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    L1 = np.random.default_rng(42).normal(0, 1, 100)
    Ln = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_3(C, where, L, L1, Ln, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm093_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    L1 = np.random.default_rng(42).normal(0, 1, 100)
    Ln = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_3(C, where, L, L1, Ln, T)
    assert isinstance(result, dict)
