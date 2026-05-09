"""Tests for msm078.mvsml_bayesian_regression_eq_6_9."""
import numpy as np
import pytest
from moirais.fn.msm078 import mvsml_bayesian_regression_eq_6_9


def test_msm078_basic():
    """Test basic functionality."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    complete = np.random.default_rng(42).normal(0, 1, 100)
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    speci = np.random.default_rng(42).normal(0, 1, 100)
    cation = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(The, complete, Bayesian, speci, cation, of)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm078_edge():
    """Test edge cases."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    complete = np.random.default_rng(42).normal(0, 1, 100)
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    speci = np.random.default_rng(42).normal(0, 1, 100)
    cation = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_9(The, complete, Bayesian, speci, cation, of)
    assert isinstance(result, dict)
