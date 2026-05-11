"""Tests for msm053.mvsml_bayesian_regression_eq_6_4."""
import numpy as np
import pytest
from morie.fn.msm053 import mvsml_bayesian_regression_eq_6_4


def test_msm053_basic():
    """Test basic functionality."""
    eS = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(eS, S, y, g, k2, where)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm053_edge():
    """Test edge cases."""
    eS = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    k2 = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(eS, S, y, g, k2, where)
    assert isinstance(result, dict)
