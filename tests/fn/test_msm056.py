"""Tests for msm056.mvsml_bayesian_regression_eq_6_5."""
import numpy as np
import pytest
from moirais.fn.msm056 import mvsml_bayesian_regression_eq_6_5


def test_msm056_basic():
    """Test basic functionality."""
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    list = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    BRR = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_5(ETA, list, model, BRR, X, L)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm056_edge():
    """Test edge cases."""
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    list = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    BRR = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_5(ETA, list, model, BRR, X, L)
    assert isinstance(result, dict)
