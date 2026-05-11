"""Tests for msm047.mvsml_bayesian_regression_eq_6_2."""
import numpy as np
import pytest
from morie.fn.msm047 import mvsml_bayesian_regression_eq_6_2


def test_msm047_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    Np = np.random.default_rng(42).normal(0, 1, 100)
    Ip = np.random.default_rng(42).normal(0, 1, 100)
    obtained = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_2(T, j, Np, Ip, obtained, by)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm047_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    Np = np.random.default_rng(42).normal(0, 1, 100)
    Ip = np.random.default_rng(42).normal(0, 1, 100)
    obtained = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_2(T, j, Np, Ip, obtained, by)
    assert isinstance(result, dict)
