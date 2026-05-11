"""Tests for msm096.mvsml_bayesian_regression_eq_6_7."""
import numpy as np
import pytest
from morie.fn.msm096 import mvsml_bayesian_regression_eq_6_7


def test_msm096_basic():
    """Test basic functionality."""
    collected = np.random.default_rng(42).normal(0, 1, 100)
    GID = np.random.default_rng(42).normal(0, 1, 100)
    Lines = np.random.default_rng(42).normal(0, 1, 100)
    individuals = np.random.default_rng(42).normal(0, 1, 100)
    Env = np.random.default_rng(42).normal(0, 1, 100)
    Environment = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_7(collected, GID, Lines, individuals, Env, Environment)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm096_edge():
    """Test edge cases."""
    collected = np.random.default_rng(42).normal(0, 1, 100)
    GID = np.random.default_rng(42).normal(0, 1, 100)
    Lines = np.random.default_rng(42).normal(0, 1, 100)
    individuals = np.random.default_rng(42).normal(0, 1, 100)
    Env = np.random.default_rng(42).normal(0, 1, 100)
    Environment = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_7(collected, GID, Lines, individuals, Env, Environment)
    assert isinstance(result, dict)
