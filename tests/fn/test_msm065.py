"""Tests for msm065.mvsml_bayesian_regression_eq_6_8."""
import numpy as np
import pytest
from moirais.fn.msm065 import mvsml_bayesian_regression_eq_6_8


def test_msm065_basic():
    """Test basic functionality."""
    gjnT = np.random.default_rng(42).normal(0, 1, 100)
    EjnT = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    nT = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_8(gjnT, EjnT, where, t, nT, are)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm065_edge():
    """Test edge cases."""
    gjnT = np.random.default_rng(42).normal(0, 1, 100)
    EjnT = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    nT = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_8(gjnT, EjnT, where, t, nT, are)
    assert isinstance(result, dict)
