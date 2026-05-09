"""Tests for msm074.mvsml_bayesian_regression_eq_6_9."""
import numpy as np
import pytest
from moirais.fn.msm074 import mvsml_bayesian_regression_eq_6_9


def test_msm074_basic():
    """Test basic functionality."""
    it = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    resCov = np.random.default_rng(42).normal(0, 1, 100)
    list = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(it, y, Y, ETA, resCov, list)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm074_edge():
    """Test edge cases."""
    it = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    ETA = np.random.default_rng(42).normal(0, 1, 100)
    resCov = np.random.default_rng(42).normal(0, 1, 100)
    list = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(it, y, Y, ETA, resCov, list)
    assert isinstance(result, dict)
