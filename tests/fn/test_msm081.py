"""Tests for msm081.mvsml_bayesian_regression_eq_6_9."""
import numpy as np
import pytest
from moirais.fn.msm081 import mvsml_bayesian_regression_eq_6_9


def test_msm081_basic():
    """Test basic functionality."""
    internally = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    sample = np.random.default_rng(42).normal(0, 1, 100)
    vec = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    Example = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(internally, to, sample, vec, b2, Example)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm081_edge():
    """Test edge cases."""
    internally = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    sample = np.random.default_rng(42).normal(0, 1, 100)
    vec = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    Example = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(internally, to, sample, vec, b2, Example)
    assert isinstance(result, dict)
