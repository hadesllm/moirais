"""Tests for msm071.mvsml_bayesian_regression_eq_6_9."""
import numpy as np
import pytest
from morie.fn.msm071 import mvsml_bayesian_regression_eq_6_9


def test_msm071_basic():
    """Test basic functionality."""
    marker = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    df0 = np.random.default_rng(42).normal(0, 1, 100)
    vT = np.random.default_rng(42).normal(0, 1, 100)
    S0 = np.random.default_rng(42).normal(0, 1, 100)
    ST = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(marker, information, df0, vT, S0, ST)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm071_edge():
    """Test edge cases."""
    marker = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    df0 = np.random.default_rng(42).normal(0, 1, 100)
    vT = np.random.default_rng(42).normal(0, 1, 100)
    S0 = np.random.default_rng(42).normal(0, 1, 100)
    ST = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(marker, information, df0, vT, S0, ST)
    assert isinstance(result, dict)
