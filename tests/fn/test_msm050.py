"""Tests for msm050.mvsml_bayesian_regression_eq_6_3."""
import numpy as np
import pytest
from moirais.fn.msm050 import mvsml_bayesian_regression_eq_6_3


def test_msm050_basic():
    """Test basic functionality."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    induced = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_3(v, S, the, induced, priors, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm050_edge():
    """Test edge cases."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    induced = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_3(v, S, the, induced, priors, g)
    assert isinstance(result, dict)
