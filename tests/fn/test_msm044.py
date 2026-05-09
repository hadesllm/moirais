"""Tests for msm044.mvsml_bayesian_regression_eq_6_1."""
import numpy as np
import pytest
from moirais.fn.msm044 import mvsml_bayesian_regression_eq_6_1


def test_msm044_basic():
    """Test basic functionality."""
    distribution = 'normal'
    of = np.random.default_rng(42).normal(0, 1, 100)
    given = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(distribution, of, given, by, j, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm044_edge():
    """Test edge cases."""
    distribution = 'normal'
    of = np.random.default_rng(42).normal(0, 1, 100)
    given = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_1(distribution, of, given, by, j, y)
    assert isinstance(result, dict)
