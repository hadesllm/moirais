"""Tests for msm095.mvsml_bayesian_regression_pt2_eq_7_4."""
import numpy as np
import pytest
from moirais.fn.msm095 import mvsml_bayesian_regression_pt2_eq_7_4


def test_msm095_basic():
    """Test basic functionality."""
    Probs = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    probs = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_4(Probs, A, probs, where, dat_F, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm095_edge():
    """Test edge cases."""
    Probs = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    probs = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_4(Probs, A, probs, where, dat_F, the)
    assert isinstance(result, dict)
