"""Tests for msm066.mvsml_bayesian_regression_eq_6_8."""
import numpy as np
import pytest
from morie.fn.msm066 import mvsml_bayesian_regression_eq_6_8


def test_msm066_basic():
    """Test basic functionality."""
    suppose = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    vec = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    N = 100
    marginally = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_8(suppose, that, vec, B, N, marginally)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm066_edge():
    """Test edge cases."""
    suppose = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    vec = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    N = 100
    marginally = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_8(suppose, that, vec, B, N, marginally)
    assert isinstance(result, dict)
